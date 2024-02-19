from langchain_community.llms import Ollama
from langchain.utilities import SQLDatabase
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from dotenv import load_dotenv
import os

question1 = 'which alarm occurred the most?'
question1 = 'hangi alarm numarası en çok gerçekleşmiştir'

model_directory = "./models/mbart-large-50-many-to-many-mmt-model"
tokenizer_directory = "./models/mbart-large-50-many-to-many-mmt-tokenizer"

model = MBartForConditionalGeneration.from_pretrained(model_directory)
tokenizer = MBart50TokenizerFast.from_pretrained(tokenizer_directory, src_lang="tr_TR")

model_inputs = tokenizer(question1, return_tensors="pt")

generated_tokens = model.generate(
    **model_inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
)

tr_to_eng = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

model_id = "mixtral-8x7b-bloke:latest"
llm = Ollama(model=model_id)

host = os.getenv('DB_HOST')
username = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
db_name = os.getenv('DB_NAME')
port = os.getenv('DB_PORT')

pgurl = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{db_name}"
db = SQLDatabase.from_uri(pgurl)
db.get_usable_table_names()

from langchain_core.prompts import ChatPromptTemplate

template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}


Question: {question}
SQL Query:"""
prompt = ChatPromptTemplate.from_template(template)

def get_schema(_):
    return db.get_table_info()

def run_query(query):
    return db.run(query)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

sql_response = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

a = sql_response.invoke({"question": tr_to_eng})

from langchain_core.prompts import ChatPromptTemplate

template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}


Question: {question}
SQL Query: {query}
SQL Response: {response}"""
prompt_response = ChatPromptTemplate.from_template(template)


full_chain = (
    RunnablePassthrough.assign(query=sql_response).assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"]),
    )
    | prompt_response
    | llm
)

b = full_chain.invoke({"question": tr_to_eng})
print(b)

model = MBartForConditionalGeneration.from_pretrained(model_directory)
tokenizer = MBart50TokenizerFast.from_pretrained(tokenizer_directory, src_lang="en_XX")

model_inputs = tokenizer(b, return_tensors="pt")

generated_tokens = model.generate(
    **model_inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id["tr_TR"]
)
eng_to_tr = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(eng_to_tr)