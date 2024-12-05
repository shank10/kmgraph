# %%
# Type hints
from typing import Any, Dict, List, Tuple, Optional

# Standard library
import ast
import logging
import re
import warnings

# Third-party packages - Data manipulation
import pandas as pd
from tqdm import tqdm

# Third-party packages - Environment & Database
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Third-party packages - Error handling & Retry logic
from tenacity import retry, stop_after_attempt, wait_exponential

# Langchain - Core
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# Langchain - Models & Connectors
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM

# Langchain - Graph & Experimental
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
#from langchain_community.graphs import Neo4jGraph
# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# api_key = os.getenv("GOOGLE_API_KEY") # if you are using Google API
movies = pd.read_csv('data/wiki_movie_plots_deduped.csv')

movies.head()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess DataFrame.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df.drop(["Wiki Page"], axis=1, inplace=True)

    # Find duplicate rows based on 'Title' column
    duplicates = df[df.duplicated(subset='Title', keep=False)]

    # Drop duplicate rows from original DataFrame
    df = df[~df['Title'].isin(duplicates['Title'])]

    # Clean string columns by stripping whitespace and replacing unknown/empty values
    # Get object columns
    col_obj = df.select_dtypes(include=["object"]).columns
    
    # Clean string columns
    for col in col_obj:
        # Strip whitespace
        df[col] = df[col].str.strip()
        
        # Replace unknown/empty values
        df[col] = df[col].apply(
            lambda x: None if pd.isna(x) or x.lower() in ["", "unknown"] 
            else x.capitalize()
        )
    
    # Drop rows with any null values
    df = df.dropna(how="any", axis=0)
    
    return df

movies = clean_data(movies).head(1000)
movies.head()

import neo4j 

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            print("Connection closed")
    
    def reset_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Database resetted successfully!")
    
    def add_document(self, documents: list):
        with self.driver.session() as session:
            for doc in documents:
                # Generate the labels string
                labels = ":".join([key for key in doc.keys()])  # Use multiple labels if needed

                # Generate properties string for Cypher query
                props_string = ", ".join(
                    f"{key}: '{value}'" if isinstance(value, str) else f"{key}: {value}"
                    for key, value in doc.items()
                )

                # Construct the Cypher CREATE query
                query = f"CREATE (n:{labels} {{{props_string}}})"
                print(query)  # For debugging purposes

                # Execute the query
                session.run(query)
   
    def execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record for record in result]
    
    def verify_connection(self):
        try:
            # Try to run a simple query
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("Connection verified successfully!")
            return True
        except Exception as e:
            print(f"Connection verification failed: {e}")
            if isinstance(e, neo4j.exceptions.AuthError):
                print("Authentication error. Please check your username and password.")
            elif isinstance(e, neo4j.exceptions.ServiceUnavailable):
                print("Could not connect to the Neo4j server. Check the server status and network configuration.")
            return False

# Connect to Neo4j
uri = "bolt://localhost:7687"
user = "neo4j"
password = "cinema123"

conn = Neo4jConnection(uri, user, password)

# Verify the connection before proceeding with any other operations
if conn.verify_connection():
    # Proceed with your operations here
    conn.add_document([{"name": "Document1", "description": "First document"}, {"name": "Document2", "description": "Second document"}])
    result = conn.execute_query("MATCH (n) RETURN n")
    print(result)
    conn.reset_database()
else:
    print("Failed to establish connection. Exiting.")

def parse_number(value: Any, target_type: type) -> Optional[float]:
    """Parse string to number with proper error handling."""
    if pd.isna(value):
        return None
    try:
        cleaned = str(value).strip().replace(',', '')
        return target_type(cleaned)
    except (ValueError, TypeError):
        return None

def clean_text(text: str) -> str:
    """Clean and normalize text fields."""
    if pd.isna(text):
        return ""
    return str(text).strip().title()

def load_movies_to_neo4j(movies_df: pd.DataFrame, connection: GraphDatabase) -> None:
    """Load movie data into Neo4j with progress tracking and error handling."""
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Query templates
    MOVIE_QUERY = """
        MERGE (movie:Movie {title: $title})
        SET movie.year = $year,
            movie.origin = $origin,
            movie.genre = $genre,
            movie.plot = $plot
    """
    
    DIRECTOR_QUERY = """
        MATCH (movie:Movie {title: $title})
        MERGE (director:Director {name: $name})
        MERGE (director)-[:DIRECTED]->(movie)
    """
    
    ACTOR_QUERY = """
        MATCH (movie:Movie {title: $title})
        MERGE (actor:Actor {name: $name})
        MERGE (actor)-[:ACTED_IN]->(movie)
    """

    # Process each movie
    for _, row in tqdm(movies_df.iterrows(), total=len(movies_df), desc="Loading movies"):
        try:
            # Prepare movie parameters
            movie_params = {
                "title": clean_text(row["Title"]),
                "year": parse_number(row["Release Year"], int),
                "origin": clean_text(row["Origin/Ethnicity"]),
                "genre": clean_text(row["Genre"]),
                "plot": str(row["Plot"]).strip()
            }
            
            # Create movie node
            connection.execute_query(MOVIE_QUERY, parameters=movie_params)
            
            # Process directors
            for director in str(row["Director"]).split(" and "):
                director_params = {
                    "name": clean_text(director),
                    "title": movie_params["title"]
                }
                connection.execute_query(DIRECTOR_QUERY, parameters=director_params)
            
            # Process cast
            if pd.notna(row["Cast"]):
                for actor in row["Cast"].split(","):
                    actor_params = {
                        "name": clean_text(actor),
                        "title": movie_params["title"]
                    }
                    connection.execute_query(ACTOR_QUERY, parameters=actor_params)
                    
        except Exception as e:
            logger.error(f"Error loading {row['Title']}: {str(e)}")
            continue

    logger.info("Finished loading movies to Neo4j")


# Load DataFrame to Neo4j
load_movies_to_neo4j(movies, conn)

query = """
MATCH (m:Movie)-[:ACTED_IN]-(a:Actor)
RETURN m.title, a.name
LIMIT 10;
"""
conn.execute_query(query)

conn.reset_database()


llm = OllamaLLM(base_url="http://127.0.0.1:11434/", model="qwen2.5-coder:latest")

df = movies.copy()

# Step 1: Define Node Labels and Properties
node_structure = "\n".join(
    [f"{col}: {', '.join(map(str, df[col][:3]))}..." for col in df.columns]
)

print(node_structure)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_node_definition(node_def: Dict) -> bool:
    """Validates the structure of the node_def dictionary returned by the system (likely an AI model or a similar generator). The expected structure of node_def is:
    {
    "NodeLabel1": {"property1": "row['property1']", "property2": "row['property2']"},
    "NodeLabel2": {"property1": "row['property1']", "property2": "row['property2']"},
}
This function ensures that the generated node definitions are correctly formatted before they are used elsewhere. It avoids potential downstream errors 
by performing an early validation.
"""
    if not isinstance(node_def, dict):
        return False
    return all(
        isinstance(v, dict) and all(isinstance(k, str) for k in v.keys())
        for v in node_def.values()
    )

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_node_definitions(chain, structure: str, example: Dict) -> Dict[str, Dict[str, str]]:
    """Fetches and validates the node definitions using a chain (e.g., a Language Model processing pipeline), with retry logic in case of transient failures.
    1. Invoke the chain: Uses the chain object to call an external process (likely an LLM) and pass the structure (dataset description) and an example (template) as inputs.
    2. Parse response: Attempts to evaluate the returned string into a Python dictionary using ast.literal_eval(). This ensures that the returned structure is properly parsed without executing unsafe code.
    3. Validate response: Calls validate_node_definition() to ensure the response is correctly formatted.
    4. Retry on errors: Retries up to 3 times with exponential backoff if parsing or validation fails."""
    try:
        # Get response from LLM
        response = chain.invoke({"structure": structure, "example": example})
        
        # Parse response
        node_defs = ast.literal_eval(response)
        
        # Validate structure
        if not validate_node_definition(node_defs):
            raise ValueError("Invalid node definition structure")
            
        return node_defs
        
    except (ValueError, SyntaxError) as e:
        logger.error(f"Error parsing node definitions: {e}")
        raise

# Updated node definition template
node_example = {
    "NodeLabel1": {"property1": "row['property1']", "property2": "row['property2'], ..."},
    "NodeLabel2": {"property1": "row['property1']", "property2": "row['property2'], ..."},
    "NodeLabel3": {"property1": "row['property1']", "property2": "row['property2'], ..."},
}

define_nodes_prompt = PromptTemplate(
    input_variables=["example", "structure"],
    template=("""
        Analyze the dataset structure below and extract the entity labels for nodes and their properties.\n
        The node properties should be based on the dataset columns and their values.\n
        Return the result as a dictionary where the keys are the node labels and the values are the node properties.\n\n
        Example: {example}\n\n
        
        Dataset Structure:\n{structure}\n\n
              
        Make sure to include all the possible node labels and their properties.\n
        If a property can be its own node, include it as a separate node label.\n
        Please do not report triple backticks to identify a code block, just return the list of tuples.\n
        Return only the dictionary containing node labels and properties, and don't include any other text or quotation.
        
        """
    ),
)

# Execute with error handling
try:
    node_chain = define_nodes_prompt | llm

    node_definitions = get_node_definitions(node_chain, structure=node_structure, example=node_example)
    logger.info(f"Node Definitions: {node_definitions}")
except Exception as e:
    logger.error(f"Failed to get node definitions: {e}")
    raise

class RelationshipIdentifier:
    """
    Identifies relationships (edges) between nodes in a graph database.

    This class uses a language model (LLM) to analyze a dataset's structure and 
    node definitions, and it identifies relationships based on those inputs. 
    It provides validation for the relationships and implements retry logic to 
    ensure robustness in handling failures.
    """

    RELATIONSHIP_EXAMPLE = [
        ("NodeLabel1", "RelationshipLabel", "NodeLabel2"),
        ("NodeLabel1", "RelationshipLabel", "NodeLabel3"),
        ("NodeLabel2", "RelationshipLabel", "NodeLabel3"),
    ]
    """
    Example relationships:
    - Each tuple represents a relationship with:
        1. Start node label (e.g., "NodeLabel1")
        2. Relationship label (e.g., "RelationshipLabel")
        3. End node label (e.g., "NodeLabel2")
    """

    PROMPT_TEMPLATE = PromptTemplate(
        input_variables=["structure", "node_definitions", "example"],
        template="""
            Consider the following Dataset Structure:\n{structure}\n\n

            Consider the following Node Definitions:\n{node_definitions}\n\n

            Based on the dataset structure and node definitions, identify relationships (edges) between nodes.\n
            Return the relationships as a list of triples where each triple contains the start node label, relationship label, and end node label, and each triple is a tuple.\n
            Please return only the list of tuples. Please do not report triple backticks to identify a code block, just return the list of tuples.\n\n

            Example:\n{example}
        """
    )
    """
    Prompt template for relationship extraction:
    - Guides the LLM to extract relationships between nodes.
    - Uses the dataset structure, node definitions, and examples as inputs.
    - Ensures the output is formatted as a list of tuples.
    """

    def __init__(self, llm: Any, logger: logging.Logger = None):
        """
        Initializes the RelationshipIdentifier.

        Args:
            llm (Any): The language model used for processing prompts.
            logger (logging.Logger, optional): Logger instance for logging activities. 
                Defaults to a module-level logger.
        """
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.chain = self.PROMPT_TEMPLATE | self.llm

    def validate_relationships(self, relationships: List[Tuple]) -> bool:
        """
        Validate the structure of identified relationships.

        Ensures that:
        - Each relationship is a tuple of length 3.
        - All elements of each tuple are strings.

        Args:
            relationships (List[Tuple]): The list of relationships to validate.

        Returns:
            bool: True if all relationships are valid, False otherwise.
        """
        return all(
            isinstance(rel, tuple) and 
            len(rel) == 3 and 
            all(isinstance(x, str) for x in rel)
            for rel in relationships
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def identify_relationships(self, structure: str, node_definitions: Dict) -> List[Tuple]:
        """
        Identify relationships between nodes based on dataset structure and node definitions.

        Implements retry logic to handle transient errors during interaction with the LLM.

        Args:
            structure (str): The dataset structure to analyze.
            node_definitions (Dict): The node definitions as a dictionary.

        Returns:
            List[Tuple]: A list of identified relationships as tuples.

        Raises:
            Exception: If the relationships cannot be identified or validated.
        """
        try:
            response = self.chain.invoke({
                "structure": structure, 
                "node_definitions": str(node_definitions), 
                "example": str(self.RELATIONSHIP_EXAMPLE)
            })
            
            # Parse the LLM response into a Python object
            relationships = ast.literal_eval(response)
            
            # Validate the structure of the relationships
            if not self.validate_relationships(relationships):
                raise ValueError("Invalid relationship structure")
                
            self.logger.info(f"Identified {len(relationships)} relationships")
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error identifying relationships: {e}")
            raise

    def get_relationship_types(self) -> List[str]:
        """
        Extract unique relationship types from identified relationships.

        This method relies on the `identify_relationships` method to fetch relationships
        and then extracts the unique relationship labels (second element of each tuple).

        Returns:
            List[str]: A list of unique relationship types.
        """
        return list(set(rel[1] for rel in self.identify_relationships()))

# Usage Example
# Initialize the identifier with a language model instance (llm)
identifier = RelationshipIdentifier(llm=llm)

# Identify relationships using dataset structure and node definitions
relationships = identifier.identify_relationships(node_structure, node_definitions)

# Output the identified relationships
print("Relationships:", relationships)

class CypherQueryBuilder:
    """Builds Cypher queries for Neo4j graph database."""

    INPUT_EXAMPLE = """
    NodeLabel1: value1, value2
    NodeLabel2: value1, value2
    """

    EXAMPLE_CYPHER = """
    CREATE (n1:NodeLabel1 {property1: row['property1'], property2: row['property2']})
    CREATE (n2:NodeLabel2 {property1: row['property1'], property2: row['property2']})
    CREATE (n1)-[:RelationshipLabel]->(n2);
    """

    PROMPT_TEMPLATE = PromptTemplate(
        input_variables=["node_definitions", "relationships", "input", "cypher"],
        template="""
        Node Definitions:\n{node_definitions}\n\n
        Relationships:\n{relationships}\n\n
        Generate Cypher queries to create nodes and relationships using the node definitions and relationships provided.\n
        Replace placeholder values with dataset properties. Return Cypher queries as a single string with each query separated by a semicolon (;).\n
        Example Input:\n{input}\n\n
        Example Cypher Query:\n{cypher}
        """
    )

    def __init__(self, llm: Any, logger: logging.Logger = None):
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.chain = self.PROMPT_TEMPLATE | self.llm

    def validate_cypher_query(self, query: str) -> bool:
        """Validate Cypher query syntax using regex patterns."""
        try:
            patterns = [
                r'CREATE \([\w:]+ \{.*?\}\)',  # Node creation with properties
                r'\)-\[:[\w:]+\]->\(',        # Relationship syntax
                r'\{.*?\}'                   # Valid property dictionary
            ]
            if not all(re.search(pattern, query) for pattern in patterns):
                self.logger.warning(f"Regex validation failed for query: {query}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False

    def sanitize_query(self, query: str) -> str:
        """Sanitize and format Cypher query."""
        return query.strip().replace('\n', ' ').replace('  ', ' ')

    def sanitize_node_definitions(self, node_definitions: Dict) -> Dict:
        """Sanitize node definitions to ensure valid Cypher syntax."""
        sanitized = {}
        for label, properties in node_definitions.items():
            sanitized_label = label.replace(' ', '_').replace('/', '_')
            sanitized_properties = {
                k.replace(' ', '_').replace('/', '_'): v for k, v in properties.items()
            }
            sanitized[sanitized_label] = sanitized_properties
        return sanitized

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def build_queries(self, node_definitions: Dict, relationships: List) -> str:
        """Build Cypher queries with retry logic."""
        try:
            response = self.chain.invoke({
                "node_definitions": str(self.sanitize_node_definitions(node_definitions)),
                "relationships": str(relationships),
                "input": self.INPUT_EXAMPLE,
                "cypher": self.EXAMPLE_CYPHER
            })

            # Extract response if wrapped in triple backticks
            response = response.strip('`') if response.startswith('```') else response

            # Sanitize response
            queries = self.sanitize_query(response)
            self.logger.debug(f"Sanitized query: {queries}")

            # Validate queries
            if not self.validate_cypher_query(queries):
                raise ValueError("Invalid Cypher query syntax")

            self.logger.info("Successfully generated Cypher queries")
            return queries

        except Exception as e:
            self.logger.error(f"Error building Cypher queries: {e}")
            raise

    def split_queries(self, queries: str) -> List[str]:
        """Split combined Cypher queries into individual statements."""
        return [q.strip() for q in queries.split(';') if q.strip()]


# Usage Example
# Assuming `llm` is a valid LLM instance and `node_definitions` and `relationships` are defined
builder = CypherQueryBuilder(llm=llm)
cypher_queries = builder.build_queries(node_definitions, relationships)
print("Cypher Queries:", cypher_queries)


logs = ""
total_rows = len(df)

def sanitize_value(value):
    """Sanitize and return value for Cypher query."""
    if isinstance(value, str):
        return value.replace('"', '\\"')  # Escape double quotes
    return str(value)

def generate_movie_queries(row):
    """Generate Cypher queries for a single movie row."""
    try:
        movie_query = f'''
            MERGE (m:Movie {{Release_Year: "{sanitize_value(row['Release Year'])}", Title: "{sanitize_value(row['Title'])}"}})
        '''
        director_query = f'''
            MERGE (d:Director {{Name: "{sanitize_value(row['Director'])}"}})
            MERGE (m)-[:DIRECTED_BY]->(d)
        '''
        cast_query = f'''
            UNWIND split("{sanitize_value(row['Cast'])}", ",") AS actor_name
            MERGE (a:Actor {{Name: actor_name}})
            MERGE (a)-[:ACTED_IN]->(m)
        '''
        genre_query = f'''
            MERGE (g:Genre {{Type: "{sanitize_value(row['Genre'])}"}})
            MERGE (m)-[:GENRE]->(g)
        '''
        plot_query = f'''
            MERGE (p:Plot {{Description: "{sanitize_value(row['Plot'])}"}})
            MERGE (m)-[:HAS_PLOT]->(p)
        '''
        origin_query = f'''
            MERGE (oe:Origin_Ethnicity {{Ethnicity: "{sanitize_value(row['Origin/Ethnicity'])}"}})
            MERGE (m)-[:HAS_ORIGIN]->(oe)
        '''
        return [movie_query, director_query, cast_query, genre_query, plot_query, origin_query]
    except KeyError as e:
        raise ValueError(f"Missing column: {e}")

def load_movies_to_neo4j(df, conn):
    """Load movie data into Neo4j with improved readability and error handling."""
    logs = []
    total_rows = len(df)

    for index, row in tqdm(df.iterrows(), total=total_rows, desc="Loading data to Neo4j"):
        try:
            queries = generate_movie_queries(row)
            with conn.driver.session() as session:
                # Use a transaction for all queries related to a single row
                with session.begin_transaction() as tx:
                    for query in queries:
                        tx.run(query)
        except Exception as e:
            log_message = f"Error on row {index + 1}: {e}"
            logs.append(log_message)
            tqdm.write(log_message)  # Print error immediately for visibility

    if logs:
        logger.error("Errors occurred during data loading:\n" + "\n".join(logs))
    else:
        logger.info("All records loaded successfully.")

# Usage
load_movies_to_neo4j(movies, conn)

'''
for index, row in tqdm(df.iterrows(), 
                      total=total_rows,
                      desc="Loading data to Neo4j",
                      position=0,
                      leave=True):
    try:
        # Generate individual CREATE statements
        queries = [
            f'CREATE (m:Movie {{Release_Year: "{sanitize_value(row["Release Year"])}", Title: "{sanitize_value(row["Title"])}"}});',
            f'CREATE (d:Director {{Name: "{sanitize_value(row["Director"])}"}});',
            f'CREATE (c:Cast {{Name: "{sanitize_value(row["Cast"])}"}});',
            f'CREATE (g:Genre {{Type: "{sanitize_value(row["Genre"])}"}});',
            f'CREATE (p:Plot {{Description: "{sanitize_value(row["Plot"])}"}});',
            f'CREATE (oe:Origin_Ethnicity {{Ethnicity: "{sanitize_value(row["Origin/Ethnicity"])}"}});',
            f'CREATE (m)-[:Directed_by]->(d);',
            f'CREATE (m)-[:Starring]->(c);',
            f'CREATE (m)-[:Genre]->(g);',
            f'CREATE (m)-[:Plot_Description]->(p);',
            f'CREATE (m)-[:Origin_Ethnicity]->(oe);'
        ]

        # Execute each query separately
        for query in queries:
            conn.execute_query(query)
            
    except Exception as e:
        logs += f"Error on row {index+1}: {str(e)}\n"

# Display logs
print(logs)  # Uncomment to display logs
'''
query = """
MATCH p=(m:Movie)-[r]-(n)
RETURN p
LIMIT 5;
"""
conn.execute_query(query)

#conn.reset_database()
movies.head()
#print(text)

llm_transformer = LLMGraphTransformer(
    llm=llm,
)

df_sample = df.head(100) # Reduce sample size for faster processing

documents = []
for _, row in tqdm(df_sample.iterrows(), 
                   total=len(df_sample), 
                   desc="Creating documents",
                   position=0, 
                   leave=True):
    try:
        # Format text with proper line breaks
        text = ""

        for col in df.columns:
            text += f"{col}: {row[col]}\n"
        
        documents.append(Document(page_content=text))
        
    except KeyError as e:
        tqdm.write(f"Missing column: {e}")
    except Exception as e:
        tqdm.write(f"Error processing row: {e}")

import asyncio

async def async_graph():
    # Replace this with your actual async code
    graph_documents = await llm_transformer.aconvert_to_graph_documents(documents)
    print(graph_documents)
    return graph_documents

graph_documents = asyncio.run(async_graph())
print(f"Nodes:{graph_documents[0].nodes}")
print(f"Relationships:{graph_documents[0].relationships}")

graph = Neo4jGraph(url=uri, username=user, password=password, enhanced_schema=True)
graph.add_graph_documents(graph_documents)

graph.refresh_schema()

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Identify the main node, and return all the relationships and nodes connected to it.
If no properties are provided, assume the nodes have only a property id.
Please don't filter on relationships or connected nodes.

Format the query as follows:
MATCH p=(n:NodeLabel)-[r]-(m)
WHERE n.id = 'value1'
RETURN p

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

chain = GraphCypherQAChain.from_llm(
    llm, 
    graph=graph, 
    verbose=True, 
    allow_dangerous_requests=True, 
    return_intermediate_steps=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT
)

chain.run("Give me an overview of the movie titled David Copperfield.")

# %%
#conn.reset_database()
conn.close()


