from selenium import webdriver
import re
import os

driver = webdriver.Chrome()
driver.get("https://en.wikipedia.org/wiki/Wikipedia:Vital_articles")
elements = driver.find_elements_by_tag_name("a")
links = []
for element in elements:
    link = element.get_attribute("href")
    try:
        if re.match("https://en.wikipedia.org/wiki/[_a-zA-Z]+$", link):
            links.append(link)
    except TypeError:
        pass

for link in links:
    driver.get(link)
    text_sections = [element.text for element in driver.find_elements_by_tag_name("p")]
    full_text = "\n".join(text_sections)
    index = len("https://en.wikipedia.org/wiki/")
    filename = f"{link[index:]}.txt"
    with open(os.path.join("corpus", filename), "w+", encoding="utf8") as f:
        f.write(full_text)

driver.close()
