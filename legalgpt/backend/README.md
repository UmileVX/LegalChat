## Getting Started

First, setup the environment:

```
poetry install
poetry shell
```

By default, we use the OpenAI LLM (though you can customize, see `app/context.py`). As a result you need to specify an `OPENAI_API_KEY` in an .env file in this directory.

Example `.env` file:

```
OPENAI_API_KEY=<openai_api_key>
```

Second, generate the embeddings of the documents in the `./data` directory (if this folder exists - otherwise, skip this step):

```
python app/engine/generate.py
```

Third, run the development server:

```
python main.py
```

Then call the API endpoint `/api/chat` to see the result:

```
curl --location 'localhost:8000/api/chat' \
--header 'Content-Type: application/json' \
--data '{ "messages": [{ "role": "user", "content": "Hello" }] }'
```

You can start editing the API by modifying `app/api/routers/chat.py`. The endpoint auto-updates as you save the file.

Open [http://localhost:8000/docs](http://localhost:8000/docs) with your browser to see the Swagger UI of the API.

The API allows CORS for all origins to simplify development. You can change this behavior by setting the `ENVIRONMENT` environment variable to `prod`:

```
ENVIRONMENT=prod uvicorn main:app
```
