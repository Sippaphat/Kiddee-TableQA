import argparse
import json
import time

############################
# You can edit you code HERE
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.experimental.query_engine.pandas import (
    PandasInstructionParser,
)
from llama_index.llms.huggingface import HuggingFaceLLM  # Import HuggingFaceLLM
from llama_index.core import PromptTemplate
import pandas as pd

############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to execute query engine.")
    parser.add_argument(
        "--query-json", type=str, required=True, help="Path to json of quert str."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./output.jsonl",
        help="Path to output response.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/project/lt900050-ai2412/Openthai13b20EFine-tuned/SuperAI_LLM_FineTune/checkpoint",
        help="Path to model.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="/project/lt900050-ai2412/Openthai13b20EFine-tuned/SuperAI_LLM_FineTune/checkpoint",
        help="Path to tokenizer.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/project/lt900050-ai2412/Openthai13b20EFine-tuned/SuperAI_LLM_FineTune/checkpoint",
        help="Path to datatable.",
    )
    args = parser.parse_args()

    ############################
    # You can edit you code HERE
    df_dataset = pd.read_excel(args.data_dir, sheet_name='customer_support_tickets.csv', header=0)
    df = pd.read_excel(args.data_dir, sheet_name='customer_support_tickets.csv', header=0)
    
    instruction_str = (
        "1. Convert the query to executable Python code using Pandas.\n"
        "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
        "3. The code should represent a solution to the query.\n"
        "4. PRINT ONLY THE EXPRESSION.\n"
        "5. Do not quote the expression.\n"
    )

    pandas_prompt_str = (
        "You are working with a pandas dataframe in Python.\n"
        "The name of the dataframe is `df`.\n"
        "This is the example of table:\n"
        "{df_str}\n\n"
        "ตารางมีคอลัมน์ดังนี้:\n"
        "{df_col}\n\n"
        "Follow these instructions:\n"
        "{instruction_str}\n"
        "Query: {query_str}\n\n"
        "Expression:"
    )

    response_synthesis_prompt_str = (
        '''Given an input question, synthesize a response from the query results. Answer as only the short answer. Just give the result. Don't answer by repeating the question. **Don't answer the pandas instruct. You will answer base on pandas output. Give out the context from csv file that you got in its original form too. \n"
        "Query: {query_str}\n\n"
        "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
        "Pandas Output: {pandas_output}\n\n"
        "Response: '''
    )
    pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
        instruction_str=instruction_str,
        df_str=df_dataset.head(5),
        df_col=df_dataset.columns.tolist()
    )

    pandas_output_parser = PandasInstructionParser(df_dataset)

    response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)

    # Load Hugging Face model from local directory
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, device_map = 'auto')
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map = 'auto')
    llm = HuggingFaceLLM(
        tokenizer=tokenizer,
        model=model,
        device_map="auto"
    )

    qp = QP(
        modules={
            "input": InputComponent(),
            "pandas_prompt": pandas_prompt,
            "llm1": llm,
            "pandas_output_parser": pandas_output_parser,
            # "response_synthesis_prompt": response_synthesis_prompt,
            #"llm2": llm2,
        },
        verbose=True,
    )

    qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
    # Load the CSV file with questions into a DataFrame

    # Initialize a list to keep track of responses
    # Iterate through each row in the DataFrame to process the questions
    ############################
    qp.run(query_str="Load system")
    with open(args.query_json, "r") as f:
        query_json = json.load(f)
    # Reset save_dir
    with open(args.save_dir, "w") as f:
        pass

    for idx, query_str in enumerate(query_json):
        t1 = time.time()
        response =  qp.run(query_str=query_str)
        generated_text = str(response)
    
        print("Generated Text:")
        print(generated_text)
        elapsed_time = time.time() - t1
        with open(args.save_dir, "a") as f:
            json.dump(
                {
                    "idx": idx,
                    "query_str": query_str,
                    "response": str(generated_text),
                    "elapsed_time": elapsed_time,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
