#include <stack>
#include <fstream>
#include "xml.h"

// Ignore space(' ', '\n', '\r', '\t')
epResult_t epXmlParser::ignoreSpace(size_t* curIdx) {
  while (*curIdx < fileContext_->size() && std::isspace(static_cast<unsigned char>(fileContext_->at(*curIdx)))) {
      ++(*curIdx);
  }
  if (*curIdx == fileContext_->size()) {
    WARN("XML: Index excceed file buffer size, file is too big");
    return epInternalError;
  }
  return epSuccess;
}

/* Value format:
 *     +--- firstIdx
 *     |     +------- retLastIdx
 *     V     V
 * key="value"
**/
epResult_t epXmlParser::getValue(std::string const& key, std::string& value, size_t firstIdx, size_t* retLastIdx) {
  if (fileContext_->at(firstIdx) != '"') {
    WARN("XML: Value format error, pos %lu char is not expected '\"' but actual '%c' for key:%s", firstIdx,
                                                                              fileContext_->at(firstIdx), key.c_str());
    return epInternalError;
  }
  size_t curIdx = firstIdx + 1;
  // find '"'
  while (curIdx < fileContext_->size() && fileContext_->at(curIdx) != '"') ++ curIdx;
  if (curIdx - firstIdx < 2) {
    INFO(EP_GRAPH, "XML: Key %s's value empty, ignore", key.c_str());
  }
  if (curIdx - firstIdx  > MAX_STR_LEN) {
    WARN("XML: Key %s's value too long (max %d)", key.c_str(), MAX_STR_LEN);
    return epInternalError;
  }
  value = fileContext_->substr(firstIdx + 1, curIdx - (firstIdx + 1));
  *retLastIdx = curIdx;
  return epSuccess;
}

/* Token format:
 *            key[space]=[space]"value"
 *            ^                       ^
 * firstIdx---+                       +---- retLastIdx
**/
epResult_t epXmlParser::getToken(std::string& key, std::string& value, size_t firstIdx, size_t* retLastIdx) {
  size_t curIdx = firstIdx;
  // Ignore space
  EP_CHECK(ignoreSpace(&curIdx));
  size_t keyFirstIdx = curIdx;
  // Get key
  while (curIdx < fileContext_->size() &&
         !std::isspace(static_cast<unsigned char>(fileContext_->at(curIdx))) && fileContext_->at(curIdx) != '=')
    curIdx ++;
  size_t keyLen = curIdx - keyFirstIdx;
  if (keyLen == 0) {
    WARN("XML: Empty key found");
    return epInternalError;
  }
  if (keyLen > MAX_STR_LEN) {
    WARN("XML: Too long key(max %d) found", MAX_STR_LEN);
    return epInternalError;
  }
  key = fileContext_->substr(keyFirstIdx, keyLen);
  // Ignore space
  EP_CHECK(ignoreSpace(&curIdx));
  if (fileContext_->at(curIdx++) != '=') {
    WARN("XML: Not found '=' with key %s", key.c_str());
    return epInternalError;
  }
  // Ignore space
  EP_CHECK(ignoreSpace(&curIdx));
  if (fileContext_->at(curIdx) != '"') {
    WARN("XML: Unexpected value with key %s", key.c_str());
    return epInternalError;
  }
  EP_CHECK(getValue(key, value, curIdx, retLastIdx));
  return epSuccess;
}

// TODO : add comment support
/*
// Shift the 3-chars string by one char and append c at the end
#define SHIFT_APPEND(s, c) do { s[0]=s[1]; s[1]=s[2]; s[2]=c; } while(0)
epResult_t xmlSkipComment(FILE* file, char* start, char next) {
  // Start from something neutral with \0 at the end.
  char end[4] = "...";

  // Inject all trailing chars from previous reads. We don't need
  // to check for --> here because there cannot be a > in the name.
  for (int i=0; i<strlen(start); i++) SHIFT_APPEND(end, start[i]);
  SHIFT_APPEND(end, next);

  // Stop when we find "-->"
  while (strcmp(end, "-->") != 0) {
    int c;
    if (fread(&c, 1, 1, file) != 1) {
      WARN("XML Parse error : unterminated comment");
      return epInternalError;
    }
    SHIFT_APPEND(end, c);
  }
  return epSuccess;
}
*/

const char* tags[] =  {
  // Xml topology node tag
  "system", "cpu", "pci", "gcu", "lare", "nic", "net",
  // Xml topoGraph node tag
  "graphs", "graph", "channel",
};
constexpr size_t NTAG = (sizeof(tags)/sizeof(char*));

/* Tag format:
 *            [space]tag
 *            ^        ^
 * firstIdx---+        +---- retLastIdx
 */
epResult_t epXmlParser::getTag(std::string& tag, size_t firstIdx, size_t* retLastIdx, size_t* retTagIdx) {
  auto curIdx = firstIdx;
  // Ignore space
  EP_CHECK(ignoreSpace(&curIdx));
  auto tagFirstIdx = curIdx;
  // Get tag
  while (curIdx < fileContext_->size() &&
         !std::isspace(static_cast<unsigned char>(fileContext_->at(curIdx))) && fileContext_->at(curIdx) != '>') curIdx ++;
  auto tagLen = curIdx - tagFirstIdx;
  if (tagLen == 0) {
    WARN("XML: Empty tag found");
    return epInternalError;
  }
  if (tagLen > MAX_STR_LEN) {
    WARN("XML: too long tag (max %d) found", MAX_STR_LEN);
    return epInternalError;
  }
  tag = fileContext_->substr(tagFirstIdx, tagLen);
  // Tag is in tags
  size_t idx = 0;
  while (idx < NTAG && strcmp(tag.c_str(), tags[idx]) != 0) ++ idx;
  if (idx == NTAG) {
    WARN("XML: Unknown tag %s", tag.c_str());
    return epInternalError;
  }
  *retLastIdx = curIdx - 1;
  *retTagIdx = idx;

  return epSuccess;
}

/* Tag structure:
 *   type [1]:
 *            <tag key=value ...>
 *            ...
 *            </tag>
 *
 *   type [2]:
 *            <tag key=value ... />
 */
epResult_t epXmlParser::load(epXml* xml) {
  // Stack store {tagIdx, XmlNode*} for create a tree
  std::stack<std::pair<size_t, epXmlNode*>> s;
  size_t curIdx = 0;
  size_t lastIdx = -1, tagIdx = -1;
  std::string tagName;
  std::string key, value;
  do {
    // Ignore space firstly
    EP_CHECK(ignoreSpace(&curIdx));
    // First find '<'
    if (fileContext_->at(curIdx++) != '<') {
      WARN("XML : lost character '<'");
      return epInternalError;
    }

    // Ignore space before tag
    EP_CHECK(ignoreSpace(&curIdx));
    tagName = "";

    // Process </tag>
    if (fileContext_->at(curIdx) == '/') {
      curIdx ++;
      EP_CHECK(getTag(tagName, curIdx, &lastIdx, &tagIdx));
      curIdx = lastIdx + 1;
      if (s.top().first != tagIdx) {
        WARN("XML: tag %s not close", tagName.c_str());
        return epInternalError;
      }
      epXmlNode* child = s.top().second;
      s.pop();
      // Root: system, graphs
      epXmlNode* parent = nullptr;
      if (!s.empty()) {
        parent = s.top().second;
        EP_CHECK(parent->addChild(child));
      }
      child->setParent(parent);
      // Ignore space before '>'
      EP_CHECK(ignoreSpace(&curIdx));
      if (fileContext_->at(curIdx++) != '>') {
        WARN("XML: Not found '>' for tag %s", tagName.c_str());
        return epInternalError;
      }
      continue;
    }

    // Process <tag attributes ..., end with '/' or '>'
    EP_CHECK(getTag(tagName, curIdx, &lastIdx, &tagIdx));
    curIdx = lastIdx + 1;
    epXmlNode* node = nullptr;
    EP_CHECK(xml->createNode(tagName.c_str(), &node));
    epXmlAttribute* attributes = node->attributes();
    // add attributes
    while (curIdx < fileContext_->size() &&
           fileContext_->at(curIdx) != '/' && fileContext_->at(curIdx) != '>') {
      size_t attrLastIdx;
      // Ignore space beforekey
      EP_CHECK(ignoreSpace(&curIdx));
      key = "";
      value = "";
      EP_CHECK(getToken(key, value, curIdx, &attrLastIdx));
      EP_CHECK(attributes->create(key.c_str(), value.c_str()));
      curIdx = attrLastIdx + 1;
    }

    // Process "/"
    if (fileContext_->at(curIdx) == '/') {
      curIdx ++;
      auto child = node;
      epXmlNode* parent = s.top().second;
      EP_CHECK(parent->addChild(child));
      child->setParent(parent);
      // Ignore space before '>'
      EP_CHECK(ignoreSpace(&curIdx));
      if (fileContext_->at(curIdx++) != '>') {
        WARN("XML: Not found '>' for tag %s.", tagName.c_str());
        return epInternalError;
      }
      continue;
    }
    // Process ..>
    if (fileContext_->at(curIdx++) == '>') {
      EP_CHECK(ignoreSpace(&curIdx));
    }

    s.push(std::make_pair(tagIdx, node));
  } while (!s.empty() && curIdx < fileContext_->size());

  // if (s.empty() && curIdx != strlen(fileBuffer_)) {
  //   WARN("XML: Format error, more than one <> in file.");
  //   return epInternalError;
  // }

  if (!s.empty() && curIdx == fileContext_->size()) {
    WARN("XML: Format error, <> not close.");
    return epInternalError;
  }
  return epSuccess;
}

epResult_t epXmlParser::loadFromFile(const char* xmlFile, epXml* xml) {
  std::ifstream file(xmlFile, std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    WARN("XML: Unable to open %s, not loading topology.", xmlFile);
    return epInternalError;
  }

  file.seekg(0, std::ios::end);
  std::streamsize fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  if (fileSize == 0) {
    WARN("XML: empty file %s", xmlFile);
    file.close();
    return epInternalError;
  }
  fileContext_->resize(static_cast<size_t>(fileSize+1), '\0');
  if (!file.read(&((*fileContext_)[0]), fileSize)) {
    WARN("XML: Error read file %s", xmlFile);
    file.close();
    return epInternalError;
  }

  epResult_t ret = load(xml);
  file.close();
  return ret;
}

epResult_t epXmlParser::dumpRec(int indent, epXmlNode* node) {
  for (int i = 0; i < indent; i ++) {
    fileContext_->append(" ");
  }
  *fileContext_ += "<" + std::string(node->name());

  epXmlAttribute& attrs = *(node->attributes());
  for (size_t a = 0; a < attrs.size(); a ++) {
    *fileContext_ += " " + std::string(attrs[a]->key()) + "=\"" + std::string(attrs[a]->value()) + "\"";
  }
  if (node->childSize() == 0) {
    *fileContext_ += "/>\n";
  } else {
    *fileContext_ += ">\n";
    for (size_t s = 0; s < node->childSize(); s ++) {
      EP_CHECK(dumpRec(indent+2, node->childAt(s)));
    }
    for (int i = 0; i < indent; i ++) {
      fileContext_->append(" ");
    }

    *fileContext_ += "</" + std::string(node->name()) + ">\n";
  }
  return epSuccess;
}

epResult_t epXmlParser::dumpToFile(const char* xmlFile, epXml* xml) {
  EP_CHECK(dumpRec(0, xml->nodeAt(0)));

  std::string xmlFileString(xmlFile);
  std::ofstream file(xmlFileString);
  if (!file.is_open()) {
    WARN("XML: Unable to open %s, not dumping topology.", xmlFile);
    return epInternalError;
  }
  file << *fileContext_;
  file.close();
  return epSuccess;
}
