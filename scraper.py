import re
import nltk
from nltk.corpus import stopwords # https://pythonspot.com/nltk-stop-words/
from urllib.parse import urlparse, urldefrag, urljoin # https://docs.python.org/3/library/urllib.parse.html 
from bs4 import BeautifulSoup # https://beautiful-soup-4.readthedocs.io/en/latest/
from collections import defaultdict # https://docs.python.org/3/library/collections.html
from urllib.robotparser import RobotFileParser # https://docs.python.org/3/library/urllib.robotparser.html
from simhash import Simhash # https://algonotes.readthedocs.io/en/latest/Simhash.html
from hashlib import md5 # https://docs.python.org/3/library/hashlib.html

#################################
# ------ GLOBAL DATA -----------#
#################################

# Download the NLTK stopwords corpus
nltk.download('stopwords')

# Set of stopwords from NLTK
STOP_WORDS = set(stopwords.words('english'))

# Additional stop words from ranks.nl (used long list)
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

# Download the additonal stopwords
STOP_WORDS = STOP_WORDS.union(ADDITIONAL_STOP_WORDS)
# Dictionary between words and their frequencies
count_Words = defaultdict(int)
# Dictionary for number_of_words
words_In_Page = {}
# List of all unique URLs encountered during crawl
uniqueCounter = []
# Counter for total pages processed 
totalePageCounter = 0
# Store exact hashes to detect duplicates
seen_hashes = set()
# List of Simhash values to detect near-duplicates
seen_simhashes = []
# Hamming distance threshold for near-duplicates
simhash_threshhold = 5
# Cache RobotFileParser objects by domain
robot_parsers = {}

# List of file extensions to avoid
bad_URL = [
    "pdf", "ppt", "pptx", "png", "zip", "jpeg", "jpg",
    "ppsx", "war", "img", "apk", "css", "js"
]

# Allowed domains
allowed_URLS = [
    r'^([A-Za-z0-9-]+\.)*ics\.uci\.edu(/.*)?$',
    r'^([A-Za-z0-9-]+\.)*cs\.uci\.edu(/.*)?$',
    r'^([A-Za-z0-9-]+\.)*informatics\.uci\.edu(/.*)?$',
    r'^([A-Za-z0-9-]+\.)*stat\.uci\.edu(/.*)?$'
]
# Compile the domains with regex
allowed_URLS_REGEXES = [re.compile(regex) for regex in allowed_URLS]
# Capture only alphabetic words
alphanumerical_Words = re.compile(r'[a-zA-Z]+')
# Skip pages with fewer than 70 words (too low content)
min_word_threshold = 70
# Tracker for number of how many unique links discovered
runningTotal = 0

#################################
# ----------- SCRAPER ----------#
#################################

def scraper(url, resp):
    links = extract_next_links(url, resp)
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

    # Decide if we parse further
    if not decideWhetherToExtractInfo(resp, absolute_url):
        return []

    # Get raw text with Beautiful Soup
    soup = BeautifulSoup(resp.raw_response.content, "lxml")
    allText = soup.get_text()

    # Check for duplicates
    if is_duplicate(allText, absolute_url):
        print(f"[Duplicate/Trap]: {absolute_url}")
        return []

    # Tokenize text
    parsedText = checkForContent(allText)
    # Initialize word counter
    counter = 0
    word_count = all_count(parsedText, counter)

    # If word count is below threshold, consider it as low content and skip
    if word_count < min_word_threshold:
        print(f"[Low Content] Skipping {absolute_url} (only {word_count} words).")
        return []

    # Store for "longest page" 
    words_In_Page[word_count] = absolute_url

    # Extract outgoing URLs from the page
    listOfLinks = getAllUrls(soup)
    return listOfLinks

#################################
# --------- VALIDATION ---------#
#################################

def is_valid(url):
    """
    # Decide whether to crawl this url or not. 
    # If you decide to crawl it, return True; otherwise return False.
    # There are already some conditions that return False.
    try:
    # Parse the URL
    """
    # Parse the URL
    try:
        parsed = urlparse(url)

        # Only allow http or https
        if parsed.scheme not in {"http", "https"}:
            return False

        # Must match an allowed domain pattern
        if not any(r.match(parsed.netloc.lower()) for r in allowed_URLS_REGEXES):
            return False

        # Check for bad file extensions
        for ext in bad_URL:
            if ext in url.lower():
                return False

        if re.search(
            r"\.(css|js|bmp|gif|ico"
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

        # If all checks pass:
        return True

    except TypeError:
        print("TypeError for ", url)
        return False

#################################
# ------- DUPLICATION CHECK ----#
#################################
# ----- Citation ------ #
# Title: Pythonic python 1: Hamming distance
# Author: Clares Loggett
# Source: https://claresloggett.github.io/python_workshops/improved_hammingdist.html
# Obtained: I use this python resource to compare two string of equal length. The number
# of positions at which the corresponding symbols are different of the strings are compared.
# The hamming distance is the minimum number of substitutions required to change into the other.
#######################################

def compute_md5(text: str) -> str:
    """Return MD5 hex digest of text."""
    # Convert the string into UTF-8 bytes
    # Use md5 to compute the checksum for these bytes
    # Hexigest returns the check as a readable hexadecimal string
    return md5(text.encode('utf-8')).hexdigest()

def compute_simhash(text: str) -> int:
    """Compute 64-bit Simhash for the text."""
    # Convert and split text into a list of tokens
    tokens = text.lower().split()
    # Create a Simhash object from the list of the tokens
    return Simhash(tokens).value

def hamming_distance(sh1: int, sh2: int) -> int:
    """Compute Hamming distance between two 64-bit integers."""
    # XOR the two integers so that bits that are different become 1 and bits that match become 0
    x = sh1 ^ sh2
    # Convert x to its binary representation as a string
    # Return count as hamming distance
    return bin(x).count('1')

def is_duplicate(current_text: str, url: str) -> bool:
    """
    Check if page is an exact duplicate
    If not exact, check near duplicate
    """
    # Use global set/lists
    global seen_hashes, seen_simhashes

    # Exact Duplicates 
    # Compute an MD5 checksum of the current page's text
    page_md5 = compute_md5(current_text)
    # If already seen, the page is an exact duplicate
    if page_md5 in seen_hashes:
        return True
    # Record md5 for future exact duplicats
    seen_hashes.add(page_md5)

    # Near-Duplicate 
    page_sh = compute_simhash(current_text)
    # Compare this SimHash to each previously stored SimHash
    # If any are within the Hamming distance, it is a near-duplicate
    for stored_sh in seen_simhashes:
        # Get hamming distance of the two pages
        dist = hamming_distance(page_sh, stored_sh)
        if dist <= simhash_threshhold:
            # Is a near duplicate
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
    # Accept text/* pages, ignoring the 'utf-8' substring
    if not re.match(r"text/.*", content_type):
        return False

    # Check page size
    if not check_URLSize(url, resp):
        return False

    # Check robots
    if not checkRobotFile(url):
        return False

    return True

#################################
# -------- WORD COUNTING -------#
#################################

def checkForContent(allText) -> list[str]:
    """
    Strip whitespace and split on spaces,
    returns raw tokens.
    """
    return allText.strip().split()

def all_count(parsedText, counter) -> int:
    """
    Counts valid words in a list of parsed tokens
    Increments for each valid word
    """
    for word in parsedText:
        # Conver token to lowercase for processing
        word_lower = word.lower()
        # Extract only alphabetic sequences
        valid_tokens = re.findall(alphanumerical_Words, word_lower)
        for t in valid_tokens:
            # Skip short tokens and words in STOP_WORDS list
            if len(t) >= 2 and t not in STOP_WORDS:
                # Increase word's frequency
                count_Words[t] += 1
                #Increment total word counter for the page
                counter += 1
    return counter

#################################
# -------- GET ALL URLS --------#
#################################

def getAllUrls(soup) -> list:
    """
    Extract links from anchor tags, remove the fragments
    track uniqueness, return as list.
    """
    # Create a set to avoid duplicates
    noDuplicateLinks = set()
    # Keep track of how many unique links are found
    global runningTotal
    global uniqueCounter

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
            # If we have never seen the the link before, add link
            if href not in uniqueCounter:
                uniqueCounter.append(href)
                # Increment the global running total of discovered links
                runningTotal += 1
            # Add to local set to avoid duplicates in this function
            noDuplicateLinks.add(href)

    return list(noDuplicateLinks)

#################################
# --------- ROBOTS.TXT ---------#
#################################

def checkRobotFile(url) -> bool:
    """
    Parse domain and check if the crawler is allowed to crawl or not
    """
    # If domain does not exist in the scope
    domain = get_domain(url)
    if not domain:
        return True

    # Create Robot Parser
    if domain not in robot_parsers:
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
        robot_parsers[domain] = rp

    rp = robot_parsers[domain]
    return rp.can_fetch("*", url)

def get_domain(url: str) -> str:
    """
    Return scheme://netloc
    """
    parsed = urlparse(url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return ""

#################################
# -------- CHECK URL SIZE ------#
#################################

def check_URLSize(url, resp, minSize=500, maxSize=(35*1024*1024)):
    """
    Skip pages < 500 bytes or > 35 MB
    """
    # Get size of bytes of the response content
    size = len(resp.raw_response.content)
    # Skip if bytes are under 500
    if size < minSize:
        print(f"[Skipping - Too Small] {url}: {size} bytes.")
        return False
    # Skip if bytes are over 35 MB
    if size > maxSize:
        print(f"[Skipping - Too Large] {url}: {size} bytes.")
        return False
    return True

#############################
# --------- REPORT -------- #
#############################

def count_unique_pages(words_In_Page) -> int:
    """
    Return # unique pages by defragmenting URLs
    """
    # Create a set to store unique defragmented URLs
    uni_Links = set()
    # Loop over each URL in the words_In_Page map
    for link in words_In_Page.values():
        # Parse the link
        parse_Link = urlparse(link)
        # Rebuild the link without query or fragment
        uni_Link = parse_Link.scheme + "://" + parse_Link.netloc + parse_Link.path
        # Add to set
        uni_Links.add(uni_Link)
    # Return how many unique pages there are
    return len(uni_Links)

def longest_page_words(words_In_Page) -> int:
    """
    Return the maximum word count among pages in 'words_In_Page'.
    """
    # If no pages are present, return 0
    if not words_In_Page:
        return 0
    # Otherwise the key with the highest value is the max word
    return max(words_In_Page.keys())

def longest_page(numberOfWords, words_In_Page) -> str:
    """
    Return the URL that has 'numberOfWords' count.
    """
    # Return the longest page by word count
    return words_In_Page.get(numberOfWords, "NoLongestPage")

def most_common_words(count_Words) -> dict:
    """
    Sort 'count_Words' by frequency descending, return top 50 as dict
    """
    sortedList = sorted(count_Words.items(), key=lambda x: x[1], reverse=True)
    return {entry[0]: entry[1] for entry in sortedList[:50]}

def getSubDomains(words_In_Page) -> list[tuple[str,int]]:
    """
    Find subdomains under .ics.uci.edu from 'words_In_Page'.
    Return list of (subdomain_url, count_of_pages).
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

    # Sort subdomains alphabetically by subdomain name
    sorted_subdomains = sorted(subdomain_counts.items(),
                               key=lambda x: (x[0].lower(), x[1]))
    return [
        (f'https://{sub}.ics.uci.edu', len(subdomain_pages[sub]))
        for sub, _ in sorted_subdomains
    ]

def printCrawlerSummary():
    """
    Print or save the crawler summary to 'report.txt'.
    """
    with open('report.txt', 'w', encoding='utf-8') as file:
        file.write('=============== Crawler Report ===============\n\n')
        file.write("Anver Chou : 91432448\n\n")

        # Unique pages
        file.write(f'Total number of Unique Pages: {len(uniqueCounter)}\n\n')

        # Longest page
        max_words = longest_page_words(words_In_Page)
        longest_url = longest_page(max_words, words_In_Page)
        file.write(f'Longest Page: {longest_url} with {max_words} words\n\n')

        # Top 50 words
        file.write('Top 50 Most Common Words:\n')
        top_words = most_common_words(count_Words)
        for w, c in top_words.items():
            file.write(f'  {w} -> {c}\n')
        file.write('\n')

        # Subdomains
        ics_subdomains = getSubDomains(words_In_Page)
        file.write('Subdomains under ics.uci.edu:\n')
        for url_sub, c_sub in ics_subdomains:
            file.write(f'  {url_sub}, {c_sub}\n')
    # Close file
    file.close()