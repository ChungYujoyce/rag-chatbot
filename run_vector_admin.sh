CHORMA_DB_PATH=./chroma_db
# docker network creater db_network

docker run -d -p 5433:5432 \
-v ~/apps/postgres:/var/lib/postgresql/data \
-e POSTGRES_USER=vectoradmin \
-e POSTGRES_PASSWORD=password \
-e POSTGRES_DB=vdbms \
--network db_network \
postgres:14-alpine;

docker run -d -p 8000:8000 \
--network db_network \
-v ${CHORMA_DB_PATH}:/chroma_db \
my-chroma-image;

docker run -it -p 3001:3001 -p 3355:3355 -p 8288:8288 \
-e SERVER_PORT="3001" \
-e JWT_SECRET="test-jwt" \
-e INNGEST_EVENT_KEY="background_workers" \
-e INNGEST_SIGNING_KEY="test-ingest" \
-e DATABASE_CONNECTION_STRING="postgresql://vectoradmin:password@host.docker.internal:5433/vdbms"  \
-e INNGEST_LANDING_PAGE="true" \
--network db_network \
--add-host=host.docker.internal:host-gateway \
mintplexlabs/vectoradmin

