#ifndef EP_XML_H_
#define EP_XML_H_

#include <stdlib.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <memory>

#include "utils.h"
#include "debug.h"
#include "checks.h"

#define EP_TOPO_XML_VERSION "1.0"
#define EP_GRAPH_XML_VERSION "1.0"
#define EP_TOPO_XML_MAX_NODES 256
#define EP_GRAPH_XML_MAX_NODES 4096

// Compile-time
constexpr int MAX_STR_LEN     = 255;
constexpr int MAX_ATTRS       = 16;
constexpr int MAX_SUB_NODES   = 128;

template <typename T> struct format;

template<> struct format<const char*> {
  static constexpr const char* string = "%s";
};

template<> struct format<char*> {
  static constexpr const char* string = "%s";
};

template<> struct format<int> {
  static constexpr const char* string = "%d";
};
template<> struct format<uint32_t> {
  static constexpr const char* string = "%u";
};

template<> struct format<int64_t> {
  static constexpr const char* string = "%ld";
};

template<> struct format<uint64_t> {
  static constexpr const char* string = "%lu";
};

template<> struct format<bool> {
  static constexpr const char* string = "%s";
};

template<> struct format<float> {
  static constexpr const char* string = "%f";
};

template<> struct format<double> {
  static constexpr const char* string = "%lf";
};

class epXmlAttributeElement {
public:
  char* key() { return key_; }
  char* value() { return value_; }
  int asInt() const { return strtol(value_, nullptr, 0);}
  int asFloat() const { return strtof(value_, nullptr);}

  epXmlAttributeElement* setKey(const char* key) {
    // If strlen(key) bigger then MAX_STR_LEN, ignore left string.
    strncpy(key_, key, MAX_STR_LEN);
    key_[MAX_STR_LEN] = '\0';
    return this;
  }

  template<typename T, typename F=format<T> >
  epXmlAttributeElement* setValue(T value) {
    // If strlen(key) bigger then MAX_STR_LEN, ignore left string.
    snprintf((char*)value_, MAX_STR_LEN, F::string, value);
    value_[MAX_STR_LEN] = '\0';
    return this;
  }

private:
  char key_[MAX_STR_LEN + 1] = {0};
  char value_[MAX_STR_LEN + 1] = {0};
};

class epXmlAttribute {
public:
  epXmlAttribute() {}

  size_t size() const { return nAttrs_; }
  // If index i is not exist, just return nullptr directly.
  epXmlAttributeElement* operator[](int i) {
    return i < nAttrs_ ? &attrs_[i] : nullptr;
  }

  epXmlAttributeElement* find(const char* key) {
    for (int i = 0; i < nAttrs_; ++ i) {
      if (strcmp(attrs_[i].key(), key) == 0) {
        return &attrs_[i];
      }
    }
    return nullptr;
  }

  epResult_t create(epXmlAttributeElement** retAttr) {
    if (nAttrs_ >= MAX_ATTRS) {
      WARN("XML: two many attributes");
      return epInternalError;
    }
    *retAttr = &attrs_[nAttrs_ ++];
    return epSuccess;
  }

  template<typename T>
  epResult_t create(const char* key, T value) {
    epXmlAttributeElement* attr;
    EP_CHECK(create(&attr));
    attr->setKey(key)->setValue(value);
    return epSuccess;
  }
  char * attrKey(int a) {
    return attrs_[a].key();
  }
  char * attrValue(int a) {
    return attrs_[a].value();
  }

private:
  int nAttrs_ = 0;
  epXmlAttributeElement attrs_[MAX_ATTRS];
};

class epXmlNode {
public:
  epXmlNode() {
    memset(name_, 0, sizeof(name_));
    parent_ = nullptr;
    for (int i = 0; i < MAX_SUB_NODES; ++ i)
      child_[i] = nullptr;
  }
  ~epXmlNode() { }

  epXmlAttribute* attributes() { return &attributes_; }
  template<typename T>
  epResult_t setAttr(const char* key, T val) {
    epXmlAttributeElement* attrElem = attributes_.find(key);
    if (attrElem == nullptr) {
      EP_CHECK(attributes_.create(key, val));
    } else {
      attrElem->setValue(val);
    }
    return epSuccess;
  }

  size_t childSize() { return nChild_; }
  // Child at index i
  epXmlNode* childAt(int i) {
    return i < nChild_ ? child_[i] : nullptr;
  }

  epResult_t addChild(epXmlNode* node) {
    if (nChild_ >= MAX_SUB_NODES) {
      WARN("XML: two many child");
      return epInternalError;
    }
    //workaround: for some gcu300 platforms, performance may be impacted by some reason when using default channel
    //            sorted by pcie. Pci node must be sorted and inserted into parent node.
    if (strcmp(node->name(), "pci") == 0) {
      epXmlAttributeElement* dstbusId = node->attributes()->find("busid");
      int pos = 0;
      for (pos = 0;pos < nChild_;pos++) {
        if (strcmp(child_[pos]->name(), "pci") == 0) {
          epXmlAttributeElement* srcbusId = child_[pos]->attributes()->find("busid");
          if (srcbusId && dstbusId && strcmp(srcbusId->value(), dstbusId->value()) >= 0) break;
        }
      }
      for (int i = nChild_; i > pos; i--) {
        child_[i] = child_[i - 1];
      }
      child_[pos] = node;
      nChild_++;
    } else {
      child_[nChild_++] = node;
    }

    return epSuccess;
  }
  epXmlNode* findChild(const char* name) {
    for(int i = 0; i < nChild_; ++ i) {
      auto node = childAt(i);
      if (strcmp(node->name(), name) == 0) {
        return node;
      }
    }
    return nullptr;
  }
  epXmlNode* findChild(const char* name, const char* key, const char* value) {
    for(int i = 0; i < nChild_; ++i) {
      auto node = childAt(i);
      auto attr = node->attributes()->find(key);
      if (node && strcmp(node->name(), name) == 0 &&
          attr && strcmp(attr->value(), value) == 0) {
        return node;
      }
    }
    return nullptr;
  }

  const char* name() const { return name_; }
  epXmlNode* setName(const char* name) {
    strncpy(name_, name, MAX_STR_LEN);
    name_[MAX_STR_LEN] = '\0';
    return this;
  }

  epXmlNode* parent() const { return parent_; }
  epXmlNode* setParent(epXmlNode* parent) {
    parent_ = parent;
    return this;
  }
  void setNChild(int nChild) {
    nChild_ = nChild;
  }
  epResult_t convertXml(uintptr_t base, int exp) {
    // For "parent", we shift the base by 1 so that we can distinguish actual
    // NULL pointers from pointers pointing to the first node.
    if (parent_)
      setParent((epXmlNode *) (exp ? ((uintptr_t)parent_ - base + 1) : (base - 1 + (uintptr_t)parent_)));

    for (int s = 0; s < nChild_; s++) {
      child_[s] = (epXmlNode *) (exp ? ((uintptr_t)child_[s] - base) : (base + (uintptr_t)child_[s]));
    }
    return epSuccess;
  }
  char* attrKey(int a) {
    return attributes_.attrKey(a);
  }
  char* attrValue(int a) {
    return attributes_.attrValue(a);
  }
  epResult_t findNode(epXmlNode* searchNode, epXmlNode** node) {
    *node = nullptr;
    // Search for the node at the current level only.
    for (int i=0; i<nChild_; i++) {
      epXmlNode* n = child_[i];
      epXmlAttribute* myChildAttrs = n->attributes();
      epXmlAttribute* searchAttrs = searchNode->attributes();
      //&& n->type == searchNode->type
      if (strcmp(n->name(), searchNode->name()) == 0  && myChildAttrs->size() == searchAttrs->size()) {
        size_t a;
        // Ensure that all the attributes are the same.
        for (a=0; a<searchAttrs->size(); a++) {
          epXmlAttributeElement* attrElem = myChildAttrs->find(searchNode->attrKey(a));
          if (attrElem == nullptr) break;
          if (strcmp(attrElem->value(), searchNode->attrValue(a)))  break;
        }
        if (a == searchNode->attributes()->size()) {
          *node = n;
          return epSuccess;
        }
      }
    }
    return epSuccess;
  }

private:
  char name_[MAX_STR_LEN + 1];
  epXmlAttribute attributes_;
  epXmlNode* parent_;
  epXmlNode* child_[MAX_SUB_NODES];
  int nChild_;
};

class epXml;
class epXmlParser {
public:
  epXmlParser() : fileContext_(std::make_unique<std::string>("")) { }
  ~epXmlParser()=default;

  epResult_t dumpToFile(const char* xmlFile, epXml* xml);
  epResult_t loadFromFile(const char* xmlFile, epXml* xml);
private:
  epResult_t dumpRec(int indent, epXmlNode* xml);

  epResult_t ignoreSpace(size_t* curIdx);
  epResult_t getValue(std::string const& key, std::string& value, size_t firstIdx, size_t* retLastIdx);
  epResult_t getToken(std::string& key, std::string& value, size_t firstIdx, size_t* retLastIdx);
  epResult_t getTag(std::string& tag, size_t firstIdx, size_t* retLastIdx, size_t* retTagIdx);
  //epResult_t skipComment(char* start, char next);
  epResult_t load(epXml* xml);

  std::unique_ptr<std::string> fileContext_;
};

class epXml {
public:
  epXml() { nNode_ = 0; }
  ~epXml()=default;
  epXml& operator=(epXml const& obj) {
    nNode_ = obj.nNode_;
    maxNodes_ = obj.maxNodes_;
    for (int i=0;i<maxNodes_;i++) {
      nodes_[i] = obj.nodes_[i];
    }
    return *this;
  }

  epResult_t loadFromFile(const char* filePath) {
    epXmlParser parser;
    return parser.loadFromFile(filePath, this);
  }
  epResult_t dumpToFile(const char* filePath) {
    epXmlParser parser;
    return parser.dumpToFile(filePath, this);
  }

  // TODO : implement function.
  void trim();

  epResult_t createNode(epXmlNode** retNode) {
    if (nNode_ >= maxNodes_) {
      WARN("XML: two many nodes");
      return epInternalError;
    }
    *retNode = &nodes_[nNode_ ++];
    return epSuccess;
  }

  epResult_t createNode(const char* name, epXmlNode** retNode) {
    EP_CHECK(createNode(retNode));
    (*retNode)->setName(name);
    return epSuccess;
  }

  bool removeNode(epXmlNode* node);

  size_t size() const { return nNode_; }
  // XmlNode at index [i]
  epXmlNode* nodeAt(int i) {
    return i < nNode_ ? &nodes_[i] : nullptr;
  }

  epXmlNode* findNode(const char* name) {
    for(int i = 0; i < nNode_; ++ i) {
      auto node = nodeAt(i);
      if (node && strcmp(node->name(), name) == 0)
        return node;
    }
    return nullptr;
  }
  epXmlNode* findNode(const char* name, const char* key, const char* value) {
    for(int i = 0; i < nNode_; ++ i) {
      auto node = nodeAt(i);
      auto attr = node->attributes()->find(key);
      if (node && strcmp(node->name(), name) == 0 &&
          attr && strcmp(attr->value(), value) == 0) {
        return node;
      }
    }
    return nullptr;
  }
  uintptr_t getNodeBase() {
    return (uintptr_t)nodes_;
  }
  // exp == 1 -- serialize; exp == 0 -- deserialize
  epResult_t convertXml(uintptr_t base, int exp) {
    for (int n = 0; n < nNode_; n++) {
      epXmlNode *node = &nodes_[n];
      EP_CHECK(node->convertXml(base, exp));
    }
    return epSuccess;
  }
  int* getNNodePtr() {return &nNode_;}
  epResult_t xmlAddTree(epXmlNode* parent, epXmlNode* srcNode) {
    epXmlNode* dstNode = nullptr;
    EP_CHECK(createNode(&dstNode));
    *dstNode = *srcNode;
    dstNode->setParent(parent);
    if (parent) {
      EP_CHECK(parent->addChild(dstNode));
    }
    dstNode->setNChild(0);
    // Recursively copy the subtree(s)
    for (size_t i=0; i<srcNode->childSize(); i++) {
      EP_CHECK(xmlAddTree(dstNode, srcNode->childAt(i)));
    }
    return epSuccess;
  }
  epResult_t fuseXmlRecursive(epXmlNode* dstParent, epXmlNode* srcParent) {
    for (size_t i = 0; i < srcParent->childSize(); i++) {
      epXmlNode* srcNode = srcParent->childAt(i);
      epXmlNode* dstNode;
      EP_CHECK(dstParent->findNode(srcNode, &dstNode));
      if (dstNode == nullptr) {
        EP_CHECK(xmlAddTree(dstParent, srcNode));
      } else {
        EP_CHECK(fuseXmlRecursive(dstNode, srcNode));
      }
    }
    return epSuccess;
  }
  epResult_t fuseXml(epXml* src) {
    epXmlNode* topNodeDst = findNode("system");
    if (topNodeDst == nullptr) {
      xmlAddTree(nullptr, (epXmlNode*)src->getNodeBase());
      return epSuccess;
    }

    epXmlNode* topNodeSrc = src->findNode("system");
    EP_CHECK(fuseXmlRecursive(topNodeDst, topNodeSrc));
    return epSuccess;
  }

public:
  int nNode_, maxNodes_;
  epXmlNode nodes_[1];
};


__attribute__((unused))
static size_t xmlMemSize(int maxNodes) {
  return offsetof(epXml, nodes_) + sizeof(epXmlNode)*maxNodes;
}
__attribute__((unused))
static epResult_t xmlAlloc(epXml** xml, int maxNodes) {
  char* mem = nullptr;
  EP_CHECK(epCalloc(&mem, xmlMemSize(maxNodes)));
  *xml = (epXml*)mem;
  (*xml)->nNode_ = 0;
  (*xml)->maxNodes_ = maxNodes;
  return epSuccess;
}

__attribute__((unused))
static epResult_t xmlNodeGetAttr(epXmlNode* node, const char* key, const char** retStr) {
  auto attr = node->attributes()->find(key);
  if (attr) {
    *retStr = attr->value();
    return epSuccess;
  } else {
    WARN("Attribute %s of node %s not found", key, node->name());
    return epInternalError;
  }
}

__attribute__((unused))
static epResult_t xmlNodeGetAttr(epXmlNode* node, const char* key, char* retStr) {
  auto attr = node->attributes()->find(key);
  if (attr) {
    strncpy(retStr, attr->value(), MAX_STR_LEN);
    retStr[MAX_STR_LEN] = '\0';
    return epSuccess;
  } else {
    WARN("Attribute %s of node %s not found", key, node->name());
    return epInternalError;
  }
}

__attribute__((unused))
static epResult_t xmlNodeGetAttr(epXmlNode* node, const char* key, int* retValue) {
  auto attr = node->attributes()->find(key);
  if (attr) {
    *retValue = strtol(attr->value(), nullptr, 0);
    return epSuccess;
  } else {
    WARN("Attribute %s of node %s not found", key, node->name());
    return epInternalError;
  }
}

__attribute__((unused))
static epResult_t xmlNodeGetAttr(epXmlNode* node, const char* key, float* retValue) {
  auto attr = node->attributes()->find(key);
  if (attr) {
    *retValue = strtof(attr->value(), nullptr);
    return epSuccess;
  } else {
    WARN("Attribute %s of node %s not found", key, node->name());
    return epInternalError;
  }
}

__attribute__((unused))
static void xmlNodeGetAttrDefault(epXmlNode* node, const char* key, int* retValue,
                                  uint64_t defaultValue) {
  *retValue = defaultValue;
  auto attr = node->attributes()->find(key);
  if (attr) {
    *retValue = strtol(attr->value(), nullptr, 0);
  }
}

__attribute__((unused))
static void xmlNodeGetAttrDefault(epXmlNode* node, const char* key, uint64_t* retValue,
                                  uint64_t defaultValue) {
  *retValue = defaultValue;
  auto attr = node->attributes()->find(key);
  if (attr) {
    *retValue = strtol(attr->value(), nullptr, 0);
  }
}

__attribute__((unused))
static void xmlNodeGetAttrDefault(epXmlNode* node, const char* key, float* retValue,
                                  float defaultValue) {
  *retValue = defaultValue;
  auto attr = node->attributes()->find(key);
  if (attr) {
    *retValue = strtof(attr->value(), nullptr);
  }
}

__attribute__((unused))
static void xmlNodeGetAttrDefault(epXmlNode* node, const char* key, const char** retStr,
                                  const char* defaultStr) {
  *retStr = defaultStr;
  auto attr = node->attributes()->find(key);
  if (attr) {
    *retStr = attr->value();
  }
}

#endif
