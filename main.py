# usage of the TableSummarizer class

from table_summarizer import TableSummarizer

# Example of how to use the imported class
if __name__ == "__main__":
    input_txt_file = 'C:\\Users\\risha\\OneDrive\\Documents\\PdfReader\\output1\\output.txt' # path to input txt file
    output_txt_file = "C:\\Users\\risha\\OneDrive\\Documents\\PdfReader\\output2\\output2.txt" # output file path
    print("hi")
    # Initialize the summarizer with your API token
    summarizer = TableSummarizer(api_token="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")
    print("hello")
    # Process the input file and write summarized content to the output file
    summarizer.process_file(input_txt_file, output_txt_file)