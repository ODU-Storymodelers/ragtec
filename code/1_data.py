from utils.news_extraction import NewsExtraction, parse_arguments

def main():
    """Main function to run the combined scraper."""
    
    # =================================================================
    # CONFIGURATION - Edit these variables to control input/output paths
    # =================================================================
    dataset_name = "drc"
    input_file = f"../data/{dataset_name}/{dataset_name}_urls.txt"
    schema_output_file = f"../data/{dataset_name}/{dataset_name}_schema.json"
    articles_output_file = f"../data/{dataset_name}/{dataset_name}_articles_gnews.json"
    
    # Operation mode: "schema", "articles", or "both"
    mode = "both"
    # =================================================================
    
    # Parse command line arguments (these will override the above if provided)
    try:
        args = parse_arguments()
    except SystemExit:
        # If argument parsing fails (like --help), create a dummy args object
        class DummyArgs:
            dataset = None
            input_file = None
            schema_output = None
            articles_output = None
            mode = None
            data_dir = "../data"
        args = DummyArgs()
    
    # Use command line arguments if provided, otherwise use the variables above
    final_dataset = args.dataset if args.dataset else dataset_name
    final_input = args.input_file if args.input_file else input_file
    final_schema_output = args.schema_output if args.schema_output else schema_output_file
    final_articles_output = args.articles_output if args.articles_output else articles_output_file
    final_mode = args.mode if args.mode else mode
    final_data_dir = args.data_dir if hasattr(args, 'data_dir') else "../data"
    
    # Create scraper instance with the final paths
    scraper = NewsExtraction(
        dataset_name=final_dataset,
        base_data_dir=final_data_dir,
        input_file=final_input,
        schema_output_file=final_schema_output,
        articles_output_file=final_articles_output
    )
    
    # Run scraper
    try:
        results = scraper.run(mode=final_mode)
        
        if "error" in results:
            print(f"Error: {results['error']}")
            return 1
        
        print(f"\nScraping completed successfully!")
        print(f"Dataset: {final_dataset}")
        print(f"Input file: {final_input}")
        if final_mode in ["schema", "both"]:
            print(f"Schema output: {final_schema_output}")
        if final_mode in ["articles", "both"]:
            print(f"Articles output: {final_articles_output}")
        return 0
        
    except KeyboardInterrupt:
        print(f"\n\nScraping interrupted by user. Progress has been saved.")
        return 1
    except Exception as e:
        print(f"\nError during scraping: {e}")
        return 1


if __name__ == "__main__":
    exit(main())