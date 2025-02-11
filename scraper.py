import re
from urllib.parse import urlparse, urljoin #https://docs.python.org/3/library/urllib.parse.html 
from bs4 import BeautifulSoup #https://www.crummy.com/software/BeautifulSoup/bs4/doc/ 
from collections import defaultdict #https://www.geeksforgeeks.org/defaultdict-in-python/ 
from hashlib import md5 #https://www.geeksforgeeks.org/md5-hash-python/#

#Variables that will be used for tracking data
unique_pages = set() #Store unique pages to prevent re-crawling
word_freq = defaultdict(int) # Store word frequency
subdomains = defaultdict(set) #Track subdomains and their pages
longest_page = {"url": "", "count": 0} #Track the page with the most words
stop_words = set() #Stores the stopped words
content_hashes = set() #Store hashs of page content to detect near-duplciates
domain_counts = defaultdict(int) #Track page count per domain

#Load the stop words
def load_stop_words():
    try:
        #Open the stopwords.txt file for list of stopped words
        with open('stopwords.txt', 'r') as f:
            for line in f:
                #Add the stop stopped words to the set
                word = line.strip()
                stop_words.add(word)
    #Error handling if the file is not found.
    except FileNotFoundError:
        print("Stop words file not found.")
load_stop_words()

#Get hash content for near-duplicates 
def get_content_hash(text):
    #Remove extra spaces and convert text to lowercase
    clean_text = re.sub(r'\s+', ' ', text).strip().lower()
    #Generate a hash for near-duplicate detection
    return md5(clean_text.encode()).hexdigest()

def scraper(url, resp):
    """Main scraping function with trap detection and data collection"""
    links = extract_next_links(url, resp)
    valid_links = [link for link in links if is_valid(link)]

    # Handle non-200 status codes
    if resp.status != 200:
        return valid_links  # Always return valid links, even if status is not 200

    # Store the actual URL 
    scrapped_url = resp.url
    # Parse the scrapped url
    parsed_url = urlparse(scrapped_url)
    # Defrag the parsed url
    defrag_url = parsed_url._replace(fragment="").geturl()

    # Check if the URL is valid and has not already been crawled
    if is_valid(defrag_url) and defrag_url not in unique_pages:
        # Store the raw HTML content of the page
        content = resp.raw_response.content

        # Size Limitation to prevent excessive processing time and memory usage
        if len(content) > 2*1024*1024:   # 2MB
            return valid_links
        
        # Extract readable text from the page
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text()  # Store the text

        # Create a hash of the text
        content_hash = get_content_hash(text)
        # Check if the hash has already been processed
        if content_hash in content_hashes:
            return valid_links  # Skip near-duplicates
        # If not a duplicate, add the hash to track processed content
        content_hashes.add(content_hash)

        # Process all alphabetic words and convert it to lowercase
        words = re.findall(r'[a-zA-Z]+', text.lower())
        # Store the number of words
        word_count = len(words)

        # Skip empty pages
        if word_count == 0:
            return valid_links
        
        # Update with the longest page encountered
        unique_pages.add(defrag_url)  # Add the defragged URL to prevent revisiting
        # Update the longest page if the current page has more words than the previous
        if word_count > longest_page["count"]:
            longest_page["url"] = defrag_url
            longest_page["count"] = word_count
        
        # Remove common stop words 
        filtered_words = [word for word in words if word not in stop_words]
        # Iterate to count occurrences of each word
        for word in filtered_words:
            word_freq[word] += 1

        # Track subdomains 
        parsed_defrag = urlparse(defrag_url)  # Parse defragmented URL
        netloc = parsed_defrag.netloc.lower()  # Extract the domain name and make it lowercase
        # Ensure domain is part of ics.uci.edu and exclude main domain
        if netloc.endswith('.ics.uci.edu') and netloc != 'ics.uci.edu':
            subdomains[netloc].add(defrag_url)
    
    # Handle 3xx redirects
    elif 300 <= resp.status < 400:
        # If status is 300, get the Location header
        redirect_url = resp.headers.get('Location')
        # If the redirected URL is valid
        if redirect_url and is_valid(redirect_url):
            return [redirect_url]  # Return the redirected URL

    # Always return valid links
    return valid_links

def extract_next_links(url, resp):
    # Implementation required.
    # url: the URL that was used to get the page
    # resp.url: the actual url of the page
    # resp.status: the status code returned by the server. 200 is OK, you got the page. Other numbers mean that there was some kind of problem.
    # resp.error: when status is not 200, you can check the error here, if needed.
    # resp.raw_response: this is where the page actually is. More specifically, the raw_response has two parts:
    #         resp.raw_response.url: the url, again
    #         resp.raw_response.content: the content of the page!
    # Return a list with the hyperlinks (as strings) scrapped from resp.raw_response.content
    #Check to see if HTTP response was successful
    if resp.status != 200 or not resp.raw_response.content:
        #Return an empty list if the response failed
        return []
    
    #Parse the page with beautifulsoup
    soup = BeautifulSoup(resp.raw_response.content, 'html.parser')
    #List to store the parsed links
    links = []
    #Find all anchor tags by seeing if they have the href attribute
    for a_tag in soup.find_all('a', href = True):
        #Extract href (the link URL)
        href = a_tag['href']
        #Converts relative URLs (/page) into absolute URLs (https://example.com/page)
        abs_url = urljoin(resp.url, href)
        #Parse and remove url fragments
        parsed = urlparse(abs_url)
        defragged_url = parsed._replace(fragment="").geturl()
        #Append the degragged url to the links list
        links.append(defragged_url)
    return links

def is_valid(url):
    # Decide whether to crawl this url or not. 
    # If you decide to crawl it, return True; otherwise return False.
    # There are already some conditions that return False.
    try:
        #Parse the url
        parsed = urlparse(url)
        #Only allow URLs that start with http, https
        if parsed.scheme not in {"http", "https"}:
            #Reject other schemes
            return False
        
        #Allow for only these domains
        allowed_domains = ['ics.uci.edu', 'cs.uci.edu', 'informatics.uci.edu', 'stat.uci.edu']
        #Extract the domain from the URl and convert it to lowercase
        netloc = parsed.netloc.lower()
        #Check if the domain matches the allowed domains or if the domain is a subdomain
        domain_matched = any(netloc == allowed or netloc.endswith(f'.{allowed}') for allowed in allowed_domains)
        #Reject URL if it is not part of the allowed domain
        if not domain_matched:
            return False
        
        #Exclude file types
        if re.search(
            r".*\.(css|js|bmp|gif|jpe?g|ico"
            + r"|png|tiff?|mid|mp2|mp3|mp4"
            + r"|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf"
            + r"|ps|eps|tex|ppt|pptx|doc|docx|xls|xlsx|names"
            + r"|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso"
            + r"|epub|dll|cnf|tgz|sha1"
            + r"|thmx|mso|arff|rtf|jar|csv"
            + r"|rm|smil|wmv|swf|wma|zip|rar|gz)$", parsed.path.lower()):
            return False

# 
# Resource: https://chatgpt.com/share/67a7efe6-3b14-800d-9fc4-26d9d5204ec3  , Accessed: 2/8/2025
# Used to get ideas to avoid for web crawlers traps
#
        #Extract path and query from the URL and convert them to lowercase
        path = parsed.path.lower()
        query = parsed.query.lower()

        #Reject URls with excessively long query parameters
        #Usually dynamically generated spam pages
        if len(parsed.query) > 100:
            return False
        
        #Reject URls with too many nested diretories
        #This can lead to infinite loops
        if len(parsed.path.split('/')) > 10:
            return False
        
        #Reject URLs with too many numeric segements
        #Can be dynamically generated pages
        if sum(1 for seg in parsed.path.split('/') if seg.isdigit()) > 3:
            return False

        #Reject URLs with matching common pagination patterns (Blogs/Forums)
        #(/page/10, /article/345, etc..)
        if re.search(r'/(page|article|post)/\d+', path):
            return False
        
        #Limit the number of pages crawled per domain to 1000
        #Prevent excessive requests to a single website
        domain = parsed.netloc.lower()
        if domain_counts[domain] > 1000: 
            return False
        domain_counts[domain] += 1

        return True
     
    except TypeError as e:
        #Return error message if an unexpected error occurs
        print (f"Error validating URL {url}:  {str(e)}")
        #Skip the URL
        return False

#Generate report
def save_report():
    #Write to report.txt file in write mode
    with open('report.txt', 'w') as f:
        #Write the count of the total number of unique pages crawled
        f.write(f"1. Unique Pages: {len(unique_pages)}\n\n")
        #Write the URL of the longest page and its number of words
        f.write(f"2. Longest Page: {longest_page['url']} ({longest_page['count']} words)\n\n")
        
        #Sort words by frequency in descending order(x[1])
        #Sort alphabetically if two words have the same frequency(x[0])
        sorted_words = sorted(word_freq.items(), key=lambda x: (-x[1], x[0]))
        #Header for top 50 words
        f.write("3. Top 50 most commons words:\n")
        #Write the top 50 words and their frequency
        for word, count in sorted_words[:50]:
            #Print a new line between each word
            f.write(f"{word}: {count}\n")
        f.write("\n")

        #Header for subdomains
        f.write("4. ICS Subdomains:\n")
        #Filter subdomains to include only those under .ics.uci.edu
        icsSubs = {k: v for k, v in subdomains.items() if k.endswith('.ics.uci.edu')}
        #Sort subdomains alphabetically
        for sub in sorted(icsSubs):
            #Write each subdomain's URL and page count
            f.write(f"http://{sub}, {len(icsSubs[sub])}\n")
        
