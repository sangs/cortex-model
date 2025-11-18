# Neo4j Mental Model Graph


# Step 1: neo4j-episode-llmgraphtransformer.ipynb 
# Step 2: neo4j-episode-programatic-chunking-embedding.ipynb
# Step 3: neo4j-episode-loader.ipynb



#Step 4: 
Aura Professional Graph Data Science is usable on demand in Aura DB Professional

  #Step 4a: Create credentials for the API Key as per: https://neo4j.com/docs/aura/api/authentication/#_creating_credentials

  
## Documentation
GDS
https://neo4j.com/docs/graph-data-science/current/machine-learning/node-embeddings/fastrp/#algorithms-embeddings-fastrp-examples-write

//Native projection
https://neo4j.com/docs/graph-data-science/current/management-ops/graph-creation/graph-project/#_relationship_orientation


# Step 5: 
Identify autodetected technology nodes (entity):

MATCH (n:Technology)
WHERE n.embedding IS NULL 
  AND n.name IS NOT NULL
  AND n.id IS NOT NULL
RETURN count(n) AS autoIdentifiedEntities, COLLECT(n.id)

# Create name property for these identified Technology nodes:
MATCH (n:Technology)
WHERE n.embedding IS NULL
  AND n.name IS NULL
  AND n.id IS NOT NULL
SET n.name = n.id
RETURN count(n) AS nodesUpdated, COLLECT(n.id) AS updatedNodeIds

# Call process_technology_nodes_without_embeddings() to create embeddings for these Technology nodes

# Ensure enbedding is present for all Technology nodes
MATCH (n:Technology)
WHERE n.embedding IS NULL 
  AND n.name IS NOT NULL
  AND n.id IS NOT NULL
RETURN count(n) AS autoIdentifiedEntities, COLLECT(n.name)

# Step 6: 
FInd node with id 0 (Zero)

MATCH (n)
WHERE id(n) = 0
RETURN n, elementId(n), labels(n), properties(n)

#Delete this node with id 0 identified above
MATCH (n)
WHERE id(n) = 0
DETACH DELETE n
RETURN 'Node with ID 0 has been deleted.' AS Status

# Step 7:
#### Trigger Embedding creation for Episode, Topic, Concept, Technology, ReferenceLink (File: 

neo4j-episode-programatic-chunking-embedding.ipynb)
add_embeddings_to_all_nodes()

# Step 8:
Project Episodes with embedding

neo4j-cypher-queries.ipynb

# Step 9:
#### Add Comprehensive topic embedding (topic+concept+technologies)
neo4j-episode-programmatic-chunking-embedding.ipynb

# Step 10:
Project, Mutate, Mutate to create property, episodeFastRPEmbedding,  SEMANTICALLY_SIMILAR_KNN relationship and MUTATED property knn_score 
Refer to neo4j-cypher-queries.ipynb

## Test: FIle: neo4j-episode-gds-retrieval-and-mcp.ipynb
# Run ADK Agent to use the following tools to test the implementation


## How to test cortex-os-mentalmodel 
=======================================
1) Start cortex-os-mentalmodel HTTP server on local by running 
./start-http-server.sh
Prereq:
- Be in folder, neo4j-employee-graph) sangeethar@MacBookAir neo4j-employee-graph % 
- Set env, source ./.venv/bin/activate

2) Start gemini cli by running
./gemini

Details below:

Gemini CLI starts and reads ~/.gemini/settings.json
Finds toolbox MCP server configuration
Launches Toolbox via stdio (MCP protocol)
Toolbox process starts
Reads podcast-episode-tools.yaml

Notes:
Gemini CLI only supports stdio MCP servers (not direct HTTP)
Toolbox bridges: Gemini CLI ↔ Toolbox (stdio) ↔ HTTP Server (HTTP)
The HTTP server must be running on port 8000 before use
Toolbox acts as a proxy/translator between MCP stdio and HTTP
Current setup:
✅ Gemini CLI configured with Toolbox MCP server
✅ Toolbox configured to use HTTP server at localhost:8000
✅ HTTP server exposes search_episodes_gds_by_question_tool
This is the same flow as Claude Desktop. 
Toolbox translates between stdio MCP (used by Gemini CLI/Claude) and HTTP (used by your server)
