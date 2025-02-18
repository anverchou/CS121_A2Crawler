import re
import nltk
from nltk.corpus import stopwords # https://pythonspot.com/nltk-stop-words/
from urllib.parse import urlparse, urldefrag, urljoin # https://docs.python.org/3/library/urllib.parse.html 
from bs4 import BeautifulSoup # https://beautiful-soup-4.readthedocs.io/en/latest/
from collections import defaultdict # https://docs.python.org/3/library/collections.html
from urllib.robotparser import RobotFileParser # https://docs.python.org/3/library/urllib.robotparser.html
from simhash import simhash # https://algonotes.readthedocs.io/en/latest/Simhash.html
from hashlib import md5 # https://docs.python.org/3/library/hashlib.html

#
# Global Data Structures
#

# Download NLTK stopwords
nltk.download('stopwords')
# Set of stopwords from NLTK
STOP_WORDS = set(stopwords.words('english'))
# Extra stop words from ranks.nl (used long list)
ADDITIONAL_STOP_WORDS = {
    "a", "able", "about", "above", "abst", "accordance", "according", "accordingly", "across", "act", "actually", "added", "adj", "affected", "affecting", "affects", "after", "afterwards", "again", 
    "against", "ah", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", 
    "anyone", "anything", "anyway", "anyways", "anywhere", "apparently", "approximately", "are", "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "auth", "available", "away", 
    "awfully", "b", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", 
    "beside", "besides", "between", "beyond", "biol", "both", "brief", "briefly", "but", "by", "c", "ca", "came", "can", "cannot", "can't", "cause", "causes", "certain", "certainly", "co", "com", "come", "comes", 
    "contain", "containing", "contains", "could", "couldnt", "d", "date", "did", "didn't", "different", "do", "does", "doesn't", "doing", "done", "don't", "down", "downwards", "due", "during", "e", "each", "ed", 
    "edu", "effect", "eg", "eight", "eighty", "either", "else", "elsewhere", "end", "ending", "enough", "especially", "et", "et-al", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", 
    "ex", "except", "f", "far", "few", "ff", "fifth", "first", "five", "fix", "followed", "following", "follows", "for", "former", "formerly", "forth", "found", "four", "from", "further", "furthermore", "g", "gave", "get",
    "gets", "getting", "give", "given", "gives", "giving", "go", "goes", "gone", "got", "gotten", "h", "had", "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he", "hed", "hence", "her", "here", "hereafter", 
    "hereby", "herein", "heres", "hereupon", "hers", "herself", "hes", "hi", "hid", "him", "himself", "his", "hither", "home", "how", "howbeit", "however", "hundred", "i", "id", "ie", "if", "i'll", "im", "immediate", "immediately", 
    "importance", "important", "in", "inc", "indeed", "index", "information", "instead", "into", "invention", "inward", "is", "isn't", "it", "itd", "it'll", "its", "itself", "i've", "j", "just", "k", "keep", "keeps", "kept", "kg",
    "km", "know", "known", "knows", "l", "largely", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets", "like", "liked", "likely", "line", "little", "ll", "look", "looking", "looks", "ltd", "m", 
    "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "million", "miss", "ml", "more", "moreover", "most", "mostly", "mr", "mrs", "much", "mug", "must", 
    "my", "myself", "n", "na", "name", "namely", "nay", "nd", "near", "nearly", "necessarily", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "ninety", "no", "nobody", "non", "none", "nonetheless", 
    "noone", "nor", "normally", "nos", "not", "noted", "nothing", "now", "nowhere", "o", "obtain", "obtained", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "omitted", "on", "once", "one", "ones", "only", "onto", "or", 
    "ord", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "owing", "own", "p", "page", "pages", "part", "particular", "particularly", "past", "per", "perhaps", "placed", "please",
    "plus", "poorly", "possible", "possibly", "potentially", "pp", "predominantly", "present", "previously", "primarily", "probably", "promptly", "proud", "provides", "put", "q", "que", "quickly", "quite", "qv", "r", "ran", "rather", "rd", 
    "re", "readily", "really", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "respectively", "resulted", "resulting", "results", "right", "run", "s", "said", "same", "saw", 
    "say", "saying", "says", "sec", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sent", "seven", "several", "shall", "she", "shed", "she'll", "shes", "should", "shouldn't", "show", "showed", 
    "shown", "showns", "shows", "significant", "significantly", "similar", "similarly", "since", "six", "slightly", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", 
    "soon", "sorry", "specifically", "specified", "specify", "specifying", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "t", "take", "taken", "taking", "tell", "tends", 
    "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof",
    "therere", "theres", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'll", "theyre", "they've", "think", "this", "those", "thou", "though", "thoughh", "thousand", "throug", "through", "throughout", "thru", "thus", 
    "til", "tip", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "ts", "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "up", "upon", 
    "ups", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "v", "value", "various", "ve", "very", "via", "viz", "vol", "vols", "vs", "w", "want", "wants", "was", "wasnt", "way", "we", "wed", "welcome", 
    "we'll", "went", "were", "werent", "we've", "what", "whatever", "what'll", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while",
    "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "whose", "why", "widely", "willing", "wish", "with", "within", "without", "wont", "words", "world", "would", "wouldnt", "www", "x", "y", "yes", 
    "yet", "you", "youd", "you'll", "your", "youre", "yours", "yourself", "yourselves", "you've", "z", "zero"
}

# Download the additional stop words
stop_Words = stop_Words.union(ADDITIONAL_STOP_WORDS)
count_Words = defaultdict(int) # Map between words and their frequencies
words_In_Page = {} # Map for number_of_words -> URL
uniqueCounter = [] # Array of all unique URLs encounters in the crawl
totalePageCounter = 0 # Counter for total pages processed
seen_hashes = set() # Exact duplicates
seen_simhashes = [] # Store simhash values 
simhash_threshhold = 5 # Hamming distance between 5 for near duplicates
robot_parsers = {} # Robot Parser

# Avoid these file extenstions
bad_URL = ["pdf", "ppt", "pptx", "png", "zip", "jpeg", "jpg", "ppsx", "war", "img", "apk"]

#Allowed domains to crawl
allowed_URLS = [
    r'^.+\.ics\.uci\.edu(/.*)?$',
    r'^.+\.cs\.uci\.edu(/.*)?$',
    r'^.+\.informatics\.uci\.edu(/.*)?$',
    r'^.+\.stat\.uci\.edu(/.*)?$'
]
allowed_URLS_REGEXES = [re.compile(regex) for regex in allowed_URLS]

# Capture only alphabetic words
alphanumerical_Words = re.compile(r'[a-zA-Z]+')

# Threshold for low content pages (page should have over 100 words)
min_word_threshold = 100

# Unique Links Encountered
runningTotal = 0

##########################
# ------ Scraper ------- #
##########################
def scraper(url, resp):
    #Extract potential links
    links = extract_next_links(url, resp)
    # Filter links by is_valid
    return [link for link in links if is_valid(link)]

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
    
    # Convert relative URL to absolute URL
    absolute_url = urljoin(resp.url, url)
    # Decide if we need to parse the url more
    if not decideWhetherToExtractInfo(resp, absolute_url):
        return []

    # Get raw text with Beautiful Soup
    soup = BeautifulSoup(resp.raw_response.content, "lxml")
    allText = soup.get_text()

    # Check for duplicates
    if is_duplicate(alltext, absolute_url):
        print(f"[Duplicate/Trap] : {absolute_url}")
        return []

    # Token text 
    parsedText = checkForContent(allText)
    #Count how many valid words the page has
    word_count = all_Count(parsedText, 0)

    # If word count is below threshhold, consider it as low content and skip
    if word_count < min_word_threshold:
        return []

    # Store the page in the words_In_Page dictioknary
    words_In_Page[word_count] = absolute_url

    #Extract outgoing URLs from the page
    listOfLinks = getAllUrls(soup)
    return listOfLinks

##########################
# ------ Validation -----#
##########################
def is_valid(url):
    # Decide whether to crawl this url or not. 
    # If you decide to crawl it, return True; otherwise return False.
    # There are already some conditions that return False.
    try:
        # Parse the URL
        parsed = urlparse(url)
        # Only allow http/htps
        if parsed.scheme not in set(["http", "https"]):
            return False
        # Check agasint allowed domain patterns
        if not any(r.match(parsed.netloc.lower()) for r in allowed_URLS_REGEXES):
            return False
        # Check for bad file extensions
        for ext in bad_URL:
            if ext in url.lower():
                return False
            
        if re.search(
            r".*\.(css|js|bmp|gif|jpe?g|ico"
            + r"|png|tiff?|mid|mp2|mp3|mp4"
            + r"|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf"
            + r"|ps|eps|tex|ppt|pptx|doc|docx|xls|xlsx|names"
            + r"|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso"
            + r"|epub|dll|cnf|tgz|sha1"
            + r"|thmx|mso|arff|rtf|jar|csv"
            + r"|rm|smil|wmv|swf|wma|zip|rar|gz)$", 
            parsed.path.lower()
        ):
            return False

    except TypeError:
        print ("TypeError for ", parsed)
        raise

#####################################
# ----- Duplcation Detection ------ #
#####################################
# ----- Citation ------ #
# Title: Pythonic python 1: Hamming distance
# Author: Clares Loggett
# Source: https://claresloggett.github.io/python_workshops/improved_hammingdist.html
# Obtained: I use this python resource to compare two string of equal length. The number
# of positions at which the corresponding symbols are different of the strings are compared.
# The hamming distance is the minimum number of substitutions required to change into the other.
#######################################

def compute_md5(text: str) -> str:
    """
    Return the hex MD5 digest of the page text.
    """
    return md5(text.encode('utf-8')).hexdigest()

def compute_simhash(text: str) -> int:
    """
    Compute Simhash of the clean text
    """

    # Tokenization
    tokens = text.lower().split()
    # Conver to integer
    return simhash(tokens)

def hamming_distance(sh1: int, sh2: int) -> int:
    """
    Compute hamming distance between two 64-bit integers.
    """
    x = sh1 ^ sh2
    return bin(x).count('1')

def is_duplicate(current_text: str, url: str) -> bool:
    """
    Check if page is an exact duplicate
    If not exact, check near duplicate
    """
    # Use global set/lists 
    global seen_hashes, seen_simhashes

    # Exact Duplicate Check
    # Compute an MD5 checksum of the current page's text
    page_md5 = compute_md5(current_text)
    # If already seeve, the page is an exact duplicate
    if page_md5 in seen_hashes:
        # Already encountered
        return True
    # Record md5 for future exact duplicates
    seen_hashes.add(page_md5)

    # Near-Duplicate Checking
    # Compute a 64-bit SimHash value for the page's text
    page_sh = compute_simhash(current_text)
    # Compare this SimHash to each previosuly stored SimHash
    # If any are within the Hamming distance, it is a near-duplicate
    for stored_sh in seen_simhashes:
        # Get hamming distance of the two pages
        dist = hamming_distance(page_sh, stored_sh)
        if dist <= simhash_threshhold:
            #Is a near duplicate
            return True
        
    # Store new simhash if not a near-duplicate for future comparsions
    seen_simhashes.append(page_sh)
    # Page is unique
    return False

def decideWhetherToExtractInfo(resp, url) -> bool:
    """
    Decide if this page is worth extracting (status, size, type, robots)
    """

    # Must return 200 resp status
    if resp.status != 200:
        return False
    
    # Must have a valid raw_response
    if resp.raw_response is None:
        return False
    
    # Content must be text and contain 'utf-8'
    content_type = resp.raw_response.headers.get("Content-Type", "").lower()
    if not re.match(r"text/.*", content_type) or "utf-8" not in content_type:
        return False
    
    # Check for URL Size
    if not check_URLSize(url, resp):
        return False
    
    # Check for robots.txt
    if not checkRobotFile(url):
        return False

    return True

###############################
# ------ Word Counting ------ #
############################### 
def checkForContent(allText) -> list[str]:
    """
    Receives allText from a webpage
    Strips leading and trailing whitespace
    Return list of raw tokens
    """
    return allText.strip().split()

def all_count(parsedText, counter) -> int:
    """
    Counts valid words in a list of parsed tokens
    Increments for each valid word
    """
    for word in parsedText: 
        # Convert token to lowercase for processing
        word_lower = word.lower()
        # Extract only alphabetic sequences
        valid_tokens = re.findall(alphanumerical_Words, word_lower)
        for t in valid_tokens:
            # Skip short tokens and words in stop_Words list
            if len(t) >= 2 and t not in stop_Words:
                # Increase word's frequency 
                count_Words[t] += 1
                # Increment total word counter for the page
                counter += 1
    return counter

############################
# ---- get Urls -----#
############################
def getAllUrls(soup) -> list:
    """
    Extract links from anchor tags, remove the fragments
    """
    # Create a set to avoid duplicates
    noDuplicateLinks = set()
    # Keep track of how many unique links are found
    global runningTotal
    # Loop over all anchor tags in the parsed HTML
    for item in soup.findAll('a'):
        # Extract the anchor attribute
        href = item.get('href')
        if href: 
            # Use urlparse to see if there is a URL fragment
            fragment = urlparse(href).fragment
            if fragment:
                # Split the # if there is a fragment
                href = href.split("#")[0]
            # Access uniqueCounter
            global uniqueCounter 
            # If we have never seen the link before, add link
            if href not in uniqueCounter:
                uniqueCounter.append(href)
                # Increment the global running total of discovered links
                runningTotal += 1
            # Add to local set to avoid duplicates in this function
            noDuplicateLinks.add(href)
    return list(noDuplicateLinks)

#########################
#------- Robots.txt --- #
#########################

robot_parsers = {}

def checkRobotFile(url) -> bool:
    """
    Checks if the crawler is allowed to crawl or not
    """
    # If domain does not exist in the scope
    domain = get_domain(url)
    if not domain:
        return True

    # Create RobotFileParser
    rp = RobotFileParser()
    rp.set_url(urljoin(domain, "/robots.txt"))
    try:
        # Read robots.txt from domain
        rp.read()
    # Error handling
    except Exception:
        robot_parsers[domain] = rp
        # Return true to crawl
        return True
    # Check if user age can fetch this 'url'
    return rp.can_fetch("*", url)

def get_domain(url: str) -> str:
    """
    Return scheme://netloc for a given URL
    """
    parsed = urlparse(url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return ""

############################
# ---- Check URL Size -----#
############################
def check_URLSIZE(url, resp, minSize=500, maxSize = (35*1024*1024)):
    """
    Skip pages that are < 500 bytes or > 35 MB
    """
    # Get size of bytes of the response content
    size = len(resp.raw_response.content)
    #Skip if byes are under 500
    if size < minSize:
        return False
    #Skip if bytes are over 35 MB
    if size > maxSize:
        return False
    return True

########################
# ------ Report ------ #
########################
def count_unique_pages(words_In_Page) -> int:
    """
    Return the number of unique pages
    """
    # Create a set to store unique defragmented URLs
    uni_Links = set()
    # Loop over each URL in the words_In_Page map
    for Link in words_In_Page.values():
        # Parse the link
        parse_Link = urlparse(Link)
        # Rebuild the link without query or fragment
        uni_Link = parse_Link.scheme + "://" + parse_Link.netloc + parse_Link.path
        # Add to set
        uni_Links.add(uni_Link)
    # Return how many unique pages there are
    return len(uni_Links)

def longest_page_words(words_In_Page) -> int:
    """
    Return the max number of words found in any single page
    """
    # If no pages are present, return 0
    if not words_In_Page:
        return 0
    # Otherwise the key with the highest value is the max word count
    return max(words_In_Page.keys())

def longest_page(nnumberOfWords, words_In_Page) -> str:
    """
    Return the page that has the highest max word count
    """
    # Find the longest page by word count
    return words_In_Page.get(nnumberOfWords, "NoLongestPage")

def most_common_words(count_Words) -> dict:
    """
    Sorts count_Words by frequency descending
    Return the top 50 words
    """
    # Sort through count_Words by descending
    sortedList = sorted(count_Words.items(), key=lambda x: x[1], reverse=True)
    # Build dictionary of top 50 words
    return {entry[0]: entry[1] for entry in sortedList[:50]}

def getSubDomains(words_In_Page) -> list[tuple[str,int]]:
    """
    Fin subdomains for 'ics.uci.edu'
    Return a list of (subdomain_url, count_of_pages).
    """

    # Track how many times each subdomain has a page
    subdomain_counts = defaultdict(int)
    # Keep the unique pages for each subdomain
    subdomain_pages = defaultdict(set)

    # Loop over URLs in words_In_Page
    for url in words_In_Page.values():
        parsed_url = urlparse(url)
        # Check if domain ends with .ics.uci.edu
        if parsed_url.netloc.endswith('.ics.uci.edu'):
            # Extract subdomain portion
            sub = parsed_url.netloc.split(".")[0]
            # Increment subdomain count
            subdomain_counts[sub] += 1
            # Add full URL to subdomain's set of pages
            subdomain_pages[sub].add(url)
    
    #Sort subdomains alphabetically by subdomain name
    sorted_subdomains = sorted(subdomain_counts.items(), key=lambda x: (x[0].lower(), x[1]))
    # Buil a list of tuples
    return [
        (f'https://{sub}.ics.uci.edu', len(subdomain_pages[sub]))
        for sub, _ in sorted_subdomains
    ]

def printCrawlerSummary():
    """
    Print crawler summary
    """
    # Open filed for writing, using UTF-8 encoding
    file = open('report.txt', 'w', encoding='utf-8')
    file.write('=============== Crawler Report ===============\n\n')
    file.write("Anver Chou : 91432448")

    # Unique Pages
    file.write(f'Total number of Unique Pages : {len(uniqueCounter)}\n\n')

    # Longest Page
    max_words = longest_page_words(words_In_Page)
    longest_url = longest_page(max_words, words_In_Page)
    file.write(f'Longest Page: {longest_url} with {max_words} words\n\n')

    # Top 50 Words
    file.write('Top 50 Most Commons Words:\n')
    top_words = most_common_words(count_Words)
    for w, c in top_words.items():
        file.write(f'  {w} -> {c}\n')
    file.write('\n')

    # Subdomains
    ics_subdomains = getSubDomains(words_In_Page)
    file.write('Subdomains under ics.uci.edu:\n')
    for url_sub, c_sub in ics_subdomains:
        file.write(f' {url_sub}, {c_sub}\n')
    
    # Close File
    file.close()