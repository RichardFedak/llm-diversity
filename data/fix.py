import csv

def extract_goodreads_id_and_title(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            
            with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
                fieldnames = ['goodreads_book_id', 'title']
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                
                writer.writeheader()
                
                for row in reader:
                    writer.writerow({
                        'goodreads_book_id': row['goodreads_book_id'],
                        'title': row['title']
                    })
        
        print(f"New CSV file created: {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

input_file_path = "books.csv"         
output_file_path = "goodreads_titles.csv" 

extract_goodreads_id_and_title(input_file_path, output_file_path)
