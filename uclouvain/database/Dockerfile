# The database for this project is a plain postgres with the 
# [pgvector](https://github.com/pgvector/pgvector) extension enabled. The docker
# image that is used to provide this service is derived from the one officially 
# provided by the pgvector project: `pgvector/pgvector:pg17`. So please refer to
# the documentation of that project in case of trouble with the docker image.
FROM pgvector/pgvector:pg17

# Create a the required table in the database. This way our machine is ready 
# from the get go.    
COPY ./CREATE_TABLE.sql /docker-entrypoint-initdb.d/

# Launch postgrest at startup.
CMD [ "postgres" ]