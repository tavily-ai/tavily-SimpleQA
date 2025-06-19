import logging
import os
import json
import asyncio
import argparse
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
from handlers import TavilyHandler, ExaHandler, GPTRHandler, PerplexityHandler
from evaluators import CorrectnessEvaluator
from utils import PostProcessor, save_summary, load_csv_data, prepare_examples, get_output_dir, save_result

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


async def get_search_handlers(search_provider_params: Dict[str, Dict[str, Any]]):
    """Initialize search handlers based on provided parameters."""
    search_handlers = []
    
    for provider_name, params in search_provider_params.items():
        if provider_name.lower() == "tavily":
            search_handlers.append(TavilyHandler(params))
        elif provider_name.lower() == "exa":
            search_handlers.append(ExaHandler(params))
        elif provider_name.lower() == "gptr":
            search_handlers.append(GPTRHandler(params))
        elif provider_name.lower() == "perplexity":
            search_handlers.append(PerplexityHandler(params))

    return search_handlers


async def evaluate_provider(
    provider_name: str,
    search_handler,
    examples: List[Dict],
    post_processor: Optional[PostProcessor] = None,
):
    """Evaluate a single search provider on the dataset."""
    evaluator = CorrectnessEvaluator()
    
    results = []
    correct_count = 0
    
    async def process_example(example):
        nonlocal correct_count
        
        query = example["question"]
        reference_answer = example["answer"]
        index = example["index"]
        
        try:
            search_result = await search_handler.search(query)
            
            original_answer = search_result.get("answer", "")
            
            is_llm_response = search_handler.is_llm_response
            if is_llm_response:
                search_ans = original_answer
            else:
                search_ans = await search_handler.post_process(search_result)
            
            answer = post_processor.extract_answer(
                query=query, 
                is_llm_response=is_llm_response, 
                search_result=search_ans
            )
            
            # Evaluate the answer
            evaluation_result = await evaluator.evaluate(
                {"question": query},
                {"answer": answer},
                {"answer": reference_answer}
            )
            
            is_correct = evaluation_result['score'] == 1.0
            if is_correct:
                correct_count += 1

            grade = evaluation_result['value']
            
            result = {
                "index": index,
                "question": query,
                "reference_answer": reference_answer,
                "predicted_answer": answer,
                "is_correct": is_correct,
                "grade": grade,
            }
            
            results.append(result)
            logger.info(f"[{provider_name}] Q{index}: Grade - {grade}, Query: '{query}'")
            save_result(result, provider_name, output_dir)

            return result
        
        except Exception as e:
            logger.error(f"[{provider_name}] Error evaluating example {index}: {str(e)}")
            results.append({
                "index": index,
                "question": query,
                "reference_answer": reference_answer,
                "predicted_answer": "ERROR",
                "is_correct": False,
                "grade": "ERROR",
                "error": str(e)
            })
            return None
    
    tasks = [process_example(example) for example in examples]
    await asyncio.gather(*tasks)
    
    accuracy = correct_count / len(examples) if examples else 0
    accuracy = round(accuracy, 3)

    return {
        "provider": provider_name,
        "results": results,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": len(examples)
    }


async def run_evaluation(
    csv_path: str,
    search_provider_params: Dict[str, Dict[str, Any]],
    start_index: int = 0,
    end_index: Optional[int] = None,
    random_sample: Optional[int] = None,
    post_process_model: str = "gpt-4.1-mini",
    parallel: bool = True,
    output_dir: str = "results",
    rerun: bool = False,
):
    """Run the benchmark evaluation using the CSV test set.
    
    Args:
        csv_path: Path to the CSV file with questions and answers
        search_provider_params: Dictionary mapping search provider names to their parameters
        start_index: Starting index for examples (inclusive)
        end_index: Ending index for examples (exclusive), defaults to the end of the dataset
        random_sample: Number of random samples to select (overrides start_index and end_index)
        post_process_model: Model to use for post-processing
        parallel: Whether to run evaluations for search providers in parallel
        output_dir: Directory to save results
        rerun: Whether to rerun evaluation on existing results directory, output_dir must exist
    """
    try:
        # Load and prepare data from CSV
        examples = load_csv_data(csv_path, start_index, end_index, random_sample)
        examples = prepare_examples(examples, list(search_provider_params.keys()), rerun, output_dir, random_sample)
    
        # Initialize search handlers
        search_handlers = await get_search_handlers(search_provider_params)
        provider_names = list(search_provider_params.keys())

        os.makedirs(output_dir, exist_ok=True)
        
        provider_results = {}
        post_processor = PostProcessor(llm_model=post_process_model)

        if parallel:
            # Evaluate providers in parallel
            tasks = []
            for handler, provider_name in zip(search_handlers, provider_names):
                task = evaluate_provider(
                    provider_name,
                    handler,
                    examples[provider_name],
                    post_processor,
                )
                tasks.append(task)
            
            # Wait for all evaluations to complete
            results = await asyncio.gather(*tasks)
            for result in results:
                provider_name = result["provider"]
                provider_results[provider_name] = result
        else:
            # Evaluate providers sequentially
            for handler, provider_name in zip(search_handlers, provider_names):
                logger.info(f"Evaluating provider: {provider_name}")
                result = await evaluate_provider(
                    provider_name,
                    handler,
                    examples[provider_name],
                    post_processor,
                )
                provider_results[provider_name] = result
        
        save_summary(provider_results, output_dir)

        print("\n===== EVALUATION RESULTS =====")
        print(f"Dataset: {csv_path}")
        print("-----------------------------")
        for provider_name, result in provider_results.items():
            print(f"{provider_name}: {result['accuracy']:.2%} ({result['correct_count']}/{result['total_count']})")
        print("=============================\n")
        
        return provider_results
    except Exception as e:
        logger.error(f"Error running evaluation: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SimpleQA benchmark using local CSV file")
    parser.add_argument("--csv_path", default="datasets/simple_qa_test_set.csv", help="Path to CSV file with questions and answers")
    parser.add_argument("--config", default="configs/config.json", type=str, help="Path to JSON config file with provider parameters")
    parser.add_argument("--start_index", type=int, default=0, help="Starting index for examples (inclusive)")
    parser.add_argument("--end_index", type=int, default=None, help="Ending index for examples (exclusive)")
    parser.add_argument("--random_sample", type=int, default=None, help="Number of random samples to select (overrides start/end index)")
    parser.add_argument("--post_process_model", default="gpt-4.1-mini", help="Model for post-processing")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    parser.add_argument("--sequential", action="store_true", help="Run providers sequentially instead of in parallel")
    parser.add_argument("--rerun", action="store_true", help="Rerun evaluation on existing results directory, output_dir must exist")
    
    args = parser.parse_args()
    
    search_provider_params = {}
    
    if args.config:
        try:
            with open(args.config, 'r') as f:
                search_provider_params = json.load(f)
            logger.info(f"Loaded provider configuration from file: {args.config}")
        except Exception as e:
            logger.error(f"Error loading provider configuration from file: {str(e)}")
    else:
        # Default to Tavily with default settings if no configuration provided
        logger.info("No provider configuration specified, using default Tavily configuration")
        search_provider_params = {
            "tavily": {
                "depth": "advanced",
                "include_raw_content": True,
                "max_results": 10,
            }
        }
    
    output_dir = get_output_dir(args.output_dir, args.rerun)

    asyncio.run(run_evaluation(
        csv_path=args.csv_path,
        search_provider_params=search_provider_params,
        start_index=args.start_index,
        end_index=args.end_index,
        random_sample=args.random_sample,
        post_process_model=args.post_process_model,
        parallel=not args.sequential,
        output_dir=output_dir,
        rerun=args.rerun,
    ))
