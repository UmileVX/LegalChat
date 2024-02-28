# LegalGPT

ChatGPT of legal domain

## Running Instructions

### Backend

The backend instance uses PostgreSQL for vector DB.
Thus, you need to add `pg_vector` extension beforehand.

```sql
postgres=# CREATE EXTENSION vector;
```

After set up the DB, you need to add the `.env` file.

```sh
cd legalgpt/backend

# copy the .env.template
cp .env.template .env
```

Open the created file, and fill in the blanks (OpenAI API key, and postgresql url):

```
# model could be gpt-4, gpt-3.5-turbo, gpt-4-vision-preview, etc.
MODEL=gpt-4
OPENAI_API_KEY=
# For generating a connection URI, see https://docs.timescale.com/use-timescale/latest/services/create-a-service
PG_CONNECTION_STRING=
```

For dependency management, LegalGPT uses `poetry`:

```sh
cd legalgpt/backend

# install dependencies into poetry's venv
poetry install
```

Before start the inference service, we need to generate the DB first:

```sh
# run the ingestion service
poetry run python3 generate.py
```

Now, run the inference service:

```sh
# run app with poetry's venv
poetry run python3 main.py
```

We currently use python3.10 and python3.11, thus, please make sure that you use the correct python version.

### Frontend

Follow the instructions below:

```sh
cd legalgpt/frontend

# make the .env file
cp .env.template .env

# install the dependencies
npm install

# to run demo, use npm run dev
npm run dev
```
