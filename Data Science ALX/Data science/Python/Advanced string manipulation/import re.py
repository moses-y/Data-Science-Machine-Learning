import re
text = "The quick brown fox jumps over 12 lazy dogs."
# Find any word of at least 4 characters
# {4,} is a greedy match
print(re.findall(r"\b\w{4,}\b", text))
# Find 'o' followed by one or more 'g's
print(re.findall(r"og+", text))

# The string to be searched
text = "abc123"
# Regex pattern with a capturing group for one or more digits
pattern = r"(\d+)"
# Searching the text for the pattern
match = re.search(pattern, text)
# Checking if a match is found
if match:
 # Retrieving the first captured group
 captured_group = match.group(1)
 print("Captured Group:", captured_group)
else:
 print("No match found.")

#String search function
text = "Python is an amazing programming language."
pattern = "amazing"
match = re.search(pattern, text)
if match:
 print("Match found:", match.group())
 print("Match starts at position:", match.start())
else:
 print("No match found.")

# string match function
pattern = "world"
text = "Hello, world!"
# Using re.match to find pattern at the start of the text
match = re.match(pattern, text)
if match:
 print("Match found:", match.group())
else:
 print("No match found.")

# Findall() function
text = "The dates are 2023-01-01, 2024-02-02"
# Single capturing group
pattern = r"(\d{4}-\d{2}-\d{2})"
# Multiple capturing groups
pattern_multiple = r"(\d{4})-(\d{2})-(\d{2})"
dates = re.findall(pattern, text)
date_parts = re.findall(pattern_multiple, text)
print(dates) # Output: ['2023-01-01', '2024-02-02']
print(date_parts)

# FindIter Function
text = "The rain in Spain stays mainly in the plain."
pattern = r"\bS\w+"
matches = re.finditer(pattern, text)
for match in matches:
    print(f"Match: {match.group()} at Position:{match.span()}")

# Replacement String with re.sub() function
text = "I love Python. Python is great for scripting. Python can be used for data analysis."
pattern = "Python"
replacement = "Java"
# Replace all occurrences of 'Python' with 'Java'
result = re.sub(pattern, replacement, text)
print(result)

# Split function with re.split() function.
text = "Words, separated, by, commas (and some spaces)"
pattern = ",\s*" # Comma followed by zero or more spaces
result = re.split(pattern, text)
print(result)

# Compile function with re.compile() function.

# Compile the regex pattern for a simple word match
pattern = re.compile(r'hello')
# Sample text
text = "Hello, world!"
# Using the compiled pattern to search in the text
match = pattern.search(text)
if match:
    print("Match found")
else:
    print("No match found")

