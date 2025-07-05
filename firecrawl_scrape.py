import os
import json
import functools
import argparse
from dotenv import load_dotenv
from firecrawl import FirecrawlApp

import map_url
from ollama_client import OllamaClient
import utils

# config 
print = functools.partial(print, flush = True)

# ANSI color codes
class Colors:
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


# Load environment variables
load_dotenv()

selected_model = None
early_stopping = 5  # Default to 5 items


# Retrieve API keys from environment variables
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")

# Initialize the FirecrawlApp
firecrawl_app = FirecrawlApp(api_key=firecrawl_api_key)


target_url = "https://www.agnoshealth.com/forums"
objective_scrape_prompt = "Scrape forums data question and reply from the page. including [forum_url, forum_headline, forum_reply, forum_reply_count, forum_posted_timestamp]"
target_scrapped_file = "firecrawl_data.txt"

# Find the page that most likely contains the objective
def find_relevant_page_via_map(objective: str, url: str, app) -> list:
    try:
        print(f"{Colors.CYAN}Understood. The objective is: {objective}{Colors.RESET}")
        print(f"{Colors.CYAN}Initiating search on the website: {url}{Colors.RESET}")
        
        map_prompt = f"""
        The map function generates a list of URLs from a website and it accepts a search parameter. Based on the objective of: {objective}, come up with a 1-2 word search parameter that will help us find the information we need. Only respond with 1-2 words nothing else.
        """

        print(f"{Colors.YELLOW}Analyzing objective to determine optimal search parameter...{Colors.RESET}")
        # Use Ollama to get search parameter
        ollama_client = OllamaClient()
        map_result_generator = ollama_client.generate_response(
            model="gemma3:4b",
            prompt=map_prompt,
            system_prompt="You are a search parameter generator. Respond with only 1-2 words, nothing else.",
            temperature=0.1,
            stream=False
        )
        
        # Handle the response
        if hasattr(map_result_generator, '__iter__') and not isinstance(map_result_generator, str):
            map_search_parameter = ''.join(map_result_generator).strip()
        else:
            map_search_parameter = map_result_generator.strip()
            
        print(f"{Colors.GREEN}Optimal search parameter identified: {map_search_parameter}{Colors.RESET}")

        print(f"{Colors.YELLOW}Mapping website using the identified search parameter...{Colors.RESET}")
        print(f"{Colors.MAGENTA}{map_search_parameter}{Colors.RESET}")
        map_website = app.map_url(url, params={"search": map_search_parameter})
        print(f"{Colors.GREEN}Website mapping completed successfully.{Colors.RESET}")
        print(f"{Colors.GREEN}Located {len(map_website['links'])} relevant links.{Colors.RESET}")
        print(f"{Colors.MAGENTA}{map_website}{Colors.RESET}")
        return map_website["links"]
    except Exception as e:
        print(f"{Colors.RED}Error encountered during relevant page identification: {str(e)}{Colors.RESET}")
        return None
    
    
# Scrape pages and see if the objective is met, if so return in json format else return None
def extract_objective_in_top_pages(map_website, objective, app) -> list:
    try:
        # Get top 10 links from the map result
        print(f"{Colors.MAGENTA}{map_website[:10]}{Colors.RESET}")
        
        link_list = map_url.urls
        link_count = len(link_list)
        
        result_scrape_list = []
        success_process_count = 0
        failed_process_count = 0
        skip_process_count = 0
        
        print(f"{Colors.CYAN}Proceeding to analyze top {len(link_list)} links: {link_list}{Colors.RESET}")
        
        for i, link in enumerate(link_list):
            if early_stopping and len(result_scrape_list) >= early_stopping:
                print(f"Early stopping on {early_stopping} items")
                exit()
            print(f"\n\n{Colors.YELLOW}Initiating scrape of page: {link}{Colors.RESET}")
            
            print(f"\n\n## Processing[{success_process_count + failed_process_count + skip_process_count}/{link_count}]\n\n")
            
            # Scrape the page
            scrape_result = app.scrape_url(link, formats= ['markdown'])
            print(f"\n{Colors.GREEN}Page scraping completed successfully.{Colors.RESET}\n")
     
            # print(f"{Colors.MAGENTA}{scrape_result}{Colors.RESET}")
            
            # Check if objective is met
            final_prompt = f"""
            Given the following scraped content and objective, determine if the objective is met.
            If it is, extract the relevant information in a simple and concise JSON format. Use only the necessary fields and avoid nested structures if possible.
            If the objective is not met with confidence, respond with 'Objective not met'.

            Objective: {objective}
            Scraped from url: {link}
            Scraped content: {scrape_result.markdown}

            Remember:
            1. Only return JSON if you are confident the objective is fully met.
            2. Keep the JSON structure as simple and flat as possible.
            3. Do not include any explanations or markdown formatting in your response.
            """
            
            system_prompt = "You are a Supervisor of a team of Website Scraper."
            
            
            ollama_client = OllamaClient()
            result_generator = ollama_client.generate_response(
                model="gemma3:4b", 
                prompt=final_prompt, 
                system_prompt=system_prompt, 
                temperature=0.1,
                stream=False
            )

            # Handle the response (could be string or generator)
            if hasattr(result_generator, '__iter__') and not isinstance(result_generator, str):
                # It's a generator, collect all chunks
                content = ''.join(result_generator)
            else:
                # It's already a string
                content = result_generator
            
            if content.lower() != "objective not met":
                print(f"{Colors.GREEN}Objective potentially fulfilled. Relevant information identified.{Colors.RESET}")
                try:
                    content_clean = str(content).replace("```json", "").replace("```", "").strip()
                    print(f"{Colors.MAGENTA}Extracted content: {content_clean[:200]}...{Colors.RESET}")
                    
                    
                    # Write to file
                    utils.write_file(target_scrapped_file, content_clean)
                    success_process_count += 1
                    
                    if i != link_count - 1:
                        utils.write_file(target_scrapped_file, "\n\n\n")
 
                    # result_scrape_list.append(json.loads(content_clean))
                    print(f"{Colors.GREEN}Successfully parsed and stored data.{Colors.RESET}")
                    
                except json.JSONDecodeError as e:
                    print(f"{Colors.RED}Error in parsing JSON response: {e}{Colors.RESET}")
                    print(f"{Colors.YELLOW}Raw content: {content_clean[:300]}...{Colors.RESET}")
                    failed_process_count += 1
                except Exception as e:
                    print(f"{Colors.RED}Unexpected error processing response: {e}{Colors.RESET}")
                    failed_process_count += 1
            else:
                print(f"{Colors.YELLOW}Objective not met on this page. Proceeding to next link...{Colors.RESET}")
                skip_process_count += 1
                
        return result_scrape_list if result_scrape_list else None
        
    
    except Exception as e:
        print(f"{Colors.RED}Error encountered during page analysis: {str(e)}{Colors.RESET}")
        return None
    




# Main function to execute the process
def main() -> None:
    global selected_model, early_stopping
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Firecrawl scraper with model selection')
    parser.add_argument('model', nargs='?', default='gemini', choices=['gemini', 'grok'],
                        help='Model to use for scraping (default: gemini)')
    parser.add_argument('--early-stopping', type=lambda x: None if x.lower() == 'none' else int(x),
                        default=10, help='Number of items to process before stopping (0-N, or "none" for no limit)')
    args = parser.parse_args()
    # selected_model = args.model
    early_stopping = args.early_stopping
    
    # print(f"Select LLM: {selected_model}")
    print(f"Early stopping limit: {early_stopping if early_stopping is not None else 'No limit'}")
    
    print(f"{Colors.YELLOW}Initiating web crawling process...{Colors.RESET}")
    
    # get sub urls from main url
    try:
        print(f"{Colors.YELLOW}Mapping website: {target_url}{Colors.RESET}")
        map_result = firecrawl_app.map_url(target_url)
        is_map_success = map_result.success

        print(f"{Colors.CYAN}Map success: {is_map_success}{Colors.RESET}")

        # store sub urls in a list
        sub_urls = []
        if (is_map_success):
            print(f"{Colors.GREEN}Found {len(map_result.links)} links{Colors.RESET}")
            for i, link in enumerate(map_result.links):
                if link not in sub_urls:
                    sub_urls.append(link)
                    print(f"{Colors.BLUE}#{i+1} | {link}{Colors.RESET}")
        else:
            print(f"{Colors.RED}Failed to map website{Colors.RESET}")
            
        map_website = sub_urls
    except Exception as e:
        print(f"{Colors.RED}Error during website mapping: {e}{Colors.RESET}")
        map_website = []
    
    if map_website:
        print(f"{Colors.GREEN}Relevant pages identified. Proceeding with detailed analysis...{Colors.RESET}")
        # Find objective in top pages
        result = extract_objective_in_top_pages(map_website, objective_scrape_prompt, firecrawl_app)
        
        
        if result:
            print(f"{Colors.GREEN}Objective successfully fulfilled. Extracted information:{Colors.RESET}")
            print(f"{Colors.MAGENTA}{json.dumps(result, indent=2)}{Colors.RESET}")
            print(f"{Colors.GREEN}Total items extracted: {len(result)}{Colors.RESET}")
            
        else:
            print(f"{Colors.RED}Unable to fulfill the objective with the available content.{Colors.RESET}")
    else:
        print(f"{Colors.RED}No relevant pages identified. Consider refining the search parameters or trying a different website.{Colors.RESET}")
    
    print(f"\n{Colors.CYAN}Processing Summary:{Colors.RESET}")
    print(f"  • Early stopping limit: {early_stopping if early_stopping is not None else 'No limit'}")
    print(f"  • Target file: {target_scrapped_file}")
    print(f"  • Objective: {objective_scrape_prompt}")

if __name__ == "__main__":
    main()