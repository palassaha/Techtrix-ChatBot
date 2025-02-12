import fitz
import json

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def process_pdf_to_json(text):
    lines = text.split("\n")
    events = []
    current_event = None
    category = None  # Initialize category outside loop

    for line in lines:
        line = line.strip()
        
        if not line:
            continue  # Skip empty lines

        if line.startswith("Category:"):
            category = line.replace("Category:", "").strip()

        elif line.startswith("Event:"):
            if current_event:
                events.append(current_event)  # Store the previous event

            event_name = line.replace("Event:", "").strip()
            current_event = {
                "category": category,
                "event": event_name,
                "description": "",
                "rules": [],
                "date_time": "",
                "venue": "",
                "coordinator": {}
            }

        elif current_event:  # Ensure an event is defined before adding details
            if line.startswith("Description:"):
                current_event["description"] = line.replace("Description:", "").strip()

            elif line.startswith("Date & Time:"):
                current_event["date_time"] = line.replace("Date & Time:", "").strip()

            elif line.startswith("Venue:"):
                current_event["venue"] = line.replace("Venue:", "").strip()

            elif line.startswith("Coordinator:"):
                coordinator_info = line.replace("Coordinator:", "").strip().split("(")
                if len(coordinator_info) == 2:
                    current_event["coordinator"]["name"] = coordinator_info[0].strip()
                    current_event["coordinator"]["contact"] = coordinator_info[1].strip(")")

            elif line.startswith("Rules:"):
                continue  # Skip the "Rules:" label
            
            else:
                current_event["rules"].append(line)  # Add rule if no other match

    if current_event:  # Add last event to list
        events.append(current_event)

    return {"events": events}

# File paths
pdf_path = "TechFest 2025 - Event Details.pdf"
output_json_path = "techfest_intents.json"

# Process PDF and save structured data
extracted_text = extract_text_from_pdf(pdf_path)
structured_data = process_pdf_to_json(extracted_text)

with open(output_json_path, "w") as f:
    json.dump(structured_data, f, indent=4)

print("Converted PDF to JSON successfully!")
