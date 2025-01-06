
import re

def remove_contact_info(text):
    return re.sub(r'📱.*?\|.*\n', '', text)  # Remove lines with phone/email/links

def fix_line_breaks(text):
    return re.sub(r'\n+', ' ', text)  # Replace line breaks with spaces

def clean_bullets_and_spaces(text):
    return re.sub(r'●', '', re.sub(r'\s+', ' ', text))

def extract_relevant_sections(text):
    sections = re.findall(r'(Experience.*?Education)', text, re.DOTALL)  # Extract "Experience" to "Education"
    return sections[0] if sections else text
