# What is this?

This project should demonstrate the use of Jina.ai (DocArray, Flows, Executors) with BERT (embeddings) and Elasticsearch as vector database.

Vector search illustration from Elastic:
![Vector Search](vector-search-diagram-cropped-white-space.png)
(Origin: https://www.elastic.co/what-is/vector-search)

# How to

Open project in VSCode and switch to dev container when prompted.

Run `./install_requirements.sh` to install all required python packages.

Project comes with Elasticsearch and Kibana with exposed ports to the host.  
Elasticsearch: http://localhost:9200/  
Kibana: http://localhost:5601/  

# Links

- DocArray - https://docarray.jina.ai/
- DocArray supported stores - https://docarray.jina.ai/advanced/document-store/
- Elasticsearch - https://www.elastic.co/elasticsearch/
- Elasticsearch Vector Search - https://www.elastic.co/what-is/vector-search
- BERT google - https://github.com/google-research/bert
- BERT with DocArray (& weaviate DB) - https://docarray.jina.ai/advanced/document-store/weaviate/
