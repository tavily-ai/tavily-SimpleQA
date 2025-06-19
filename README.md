# **Search Providers Evaluation System**

## **Overview**
This project provides an evaluation system of [SimpleQA](https://openai.com/index/introducing-simpleqa/) benchmark for different search providers.

### **Features**
- Evaluation of different search providers
- Customizable configuration for each provider
- Parallel independent evaluation
- Continue evaluation from where it stopped if something went wrong

---

## **Evaluation Results**

The table below shows evaluation results across different search providers and LLMs on the SimpleQA benchmark. 
**NOTE**: For transparency and accuracy, we present the higher score between our internal evaluation results and officially reported scores for supported providers. For other providers, we display their publicly reported results. 

| Provider | Accuracy |
|----------|-------|
| Tavily   | 93.3%   |
| Perplexity Sonar-Pro | 88.8% |
| Exa Search ([link](https://exa.ai/blog/api-evals)) | 90.04%   |
| OpenAI Web Search ([link](https://openai.com/index/new-tools-for-building-agents/)) | 90%  |
| GPT 4.5 ([link](https://openai.com/index/introducing-gpt-4-5/#:~:text=remain%E2%80%94a%20mystery.-,Deeper,-world%20knowledge)) | 62.5%  |
| Gemini 2.5 Pro ([link](https://deepmind.google/models/gemini/#:~:text=Factuality-,SimpleQA,-50.8%25)) | 50.8%  |

---

## **Running Locally**

1. **Clone the repository**:
    ```sh
    git clone https://github.com/simpleQA-eval.git
    cd simpleQA-eval
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Set up environment variables**:  
    Create a `.env` file in the root directory and add the following:
    ```env
    TAVILY_API_KEY=XXX
    OPENAI_API_KEY=XXX
    EXA_API_KEY=XXX
    PERPLEXITY_API_KEY=XXX
    ```

4. **Run**:
```sh
python run_evaluation.py
```

### **Command Line Options**

- `--csv_path`: Path to CSV file with questions and answers (default: datasets/simple_qa_test_set.csv)
- `--config`: Path to JSON config file with provider parameters (default: configs/config.json)
- `--start_index`: Starting index for examples (inclusive, default: 0)
- `--end_index`: Ending index for examples (exclusive, default: all examples)
- `--random_sample`: Number of random samples to select (overrides start/end index)
- `--post_process_model`: Model for post-processing (default: gpt-4.1-mini)
- `--output_dir`: Directory to save results (default: results)
- `--sequential`: Run providers sequentially instead of in parallel
- `--rerun`: Continue evaluation on existing results directory, output_dir must exist

### **Config Example**

Configuration file `config.json` might look like:
```json
{
  "tavily": {
    "depth": "advanced",
    "include_raw_content": true,
    "max_results": 10,
  },
  "perplexity": {
    "model": "sonar-pro",
  }
}
```

### **Results Output**

The script generates two types of output files in the specified output directory:
- Detailed results CSV for each provider (questions, answers, and evaluation grades)
- Summary CSV with accuracy metrics for all providers

### **Continue Benchmark**

If your benchmark evaluation is interrupted, you can continue from where it stopped using the `--rerun` flag (`output_dir` folder must exists with the previous run partial results):

```sh
python run_evaluation.py --output_dir results/my_evaluation --rerun
```

This will:
1. Load existing results from the specified output directory
2. Skip questions that have already been evaluated
3. Continue with remaining questions in the dataset
4. Update the summary statistics with all results when complete

---

## **Adding a New Search Provider to the Evaluation**
### Supported Search Providers
The current supported search providers are:
- `tavily`
- `perplexity`
- `gptr`
- `exa`

You can extend the system to evaluate additional search providers by following these steps:

1. Create a new handler file in the `handlers` directory (e.g., `handlers/new_provider_handler.py`).

2. Add your provider to the handler registry:
- Update `handlers/__init__.py` to import and expose your new handler.
- Update the `get_search_handlers` function in `app.py` and `run_benchmark.py` to include your new provider.

3. Update environment variables, add your provider's API key to the `.env` file:
```
NEW_PROVIDER_API_KEY=your_api_key_here
```

4. Use your provider in evaluation config:
```json
{
  "new_provider": {
    "custom_param1": "value1",
    "custom_param2": "value2"
  }
}
```

Remember to implement appropriate error handling and respect any rate limits or API constraints for your new provider.

---