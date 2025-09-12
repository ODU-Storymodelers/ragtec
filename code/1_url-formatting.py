import csv    
import os         

def extract_urls(file_path):
    urls = []

    # Try reading the file as a CSV
    try:
        with open(file_path, "r", encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            for row in reader:
                # Each row may contain multiple URLs, so split if necessary
                for url in row:
                    clean_url = url.strip('')  # Remove extra spaces and quotes
                    if clean_url:
                        urls.append(clean_url)
    except Exception as e:
        print(f"Error reading file: {e}")

    return urls


# Loading the URLs from a CSV file
dataset_name = "sudan"  # Change this to your dataset name
file_path = f"../data/{dataset_name}/{dataset_name.capitalize()}_URL_List.csv"  # Change this to your file path
urls = extract_urls(file_path)

# Define the output file name dynamically based on the extracted country name
output_file = f"../data/{dataset_name}/{dataset_name}_urls.txt"

# Save the extracted URLs in a text file while keeping double quotes
try:
    with open(output_file, "w", encoding="utf-8") as file:
        for url in urls:
            file.write(f"{url}\n")  # Keep each URL on a new line

    print(f"Successfully saved {len(urls)} URLs to {output_file}")
except Exception as e:
    print(f"Error saving file: {e}")

# Print the first few URLs to verify they are correctly formatted
print(urls[:10])