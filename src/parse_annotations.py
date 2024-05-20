import xml.etree.ElementTree as ET

def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    
    filename = root.find('filename').text + ".jpg"
    breed = root.find('object').find('name').text
    return filename, breed
