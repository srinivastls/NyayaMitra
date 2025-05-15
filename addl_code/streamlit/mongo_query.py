from pymongo import MongoClient
# Connect to the local MongoDB instance
from bson import ObjectId

def serialize_document(doc):
    """Convert MongoDB document to a serializable format."""
    if '_id' in doc:
        doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
    return doc

client = MongoClient("mongodb://localhost:27018/")
db = client['Legal']

def pending_prayers_data(query, page_number=1, page_size=10):
    ret_object = {
        "total_count": 0,
        "pending_prayers": [],
    }
    pending_prayers_collection = db['pending_prayers']

    # Split the query into keywords
    keywords = query.split(' ')
    regex_queries = [{"search_column": {"$regex": keyword, "$options": "i"}} for keyword in keywords]

    # MongoDB query with AND logic
    query_filter = {
        "$and": regex_queries
    }

    # Calculate total count of matching documents
    total_count = pending_prayers_collection.count_documents(query_filter)
    ret_object["total_count"] = total_count

    # Apply pagination
    skip = (page_number - 1) * page_size
    prayers = pending_prayers_collection.find(query_filter).skip(skip).limit(page_size)

    for prayer in prayers:
        ret_object['pending_prayers'].append(serialize_document(prayer))

    return ret_object


def disposed_prayers_data(query, page_number=1, page_size=10):
    ret_object = {
        "total_count": 0,
        "disposed_prayers": [],
    }
    disposed_prayers_collection = db['disposed_prayers']

    # Split the query into keywords
    keywords = query.split(' ')
    regex_queries = [{"search_column": {"$regex": keyword, "$options": "i"}} for keyword in keywords]

    # MongoDB query with AND logic
    query_filter = {
        "$and": regex_queries
    }

    # Calculate total count of matching documents
    total_count = disposed_prayers_collection.count_documents(query_filter)
    ret_object["total_count"] = total_count

    # Apply pagination
    skip = (page_number - 1) * page_size
    prayers = disposed_prayers_collection.find(query_filter).skip(skip).limit(page_size)

    for prayer in prayers:
        ret_object['disposed_prayers'].append(serialize_document(prayer))

    return ret_object


def orders_data(query, page_number=1, page_size=10):
    ret_object = {
        "total_count": 0,
        "orders": [],
    }
    orders_collection = db['orders']

    # Split the query into keywords
    keywords = query.split(' ')
    regex_queries = [{"search_column": {"$regex": keyword, "$options": "i"}} for keyword in keywords]

    # MongoDB query with AND logic
    query_filter = {
        "$and": regex_queries
    }

    # Calculate total count of matching documents
    total_count = orders_collection.count_documents(query_filter)
    ret_object["total_count"] = total_count

    # Apply pagination
    skip = (page_number - 1) * page_size
    prayers = orders_collection.find(query_filter).skip(skip).limit(page_size)

    for prayer in prayers:
        ret_object['orders'].append(serialize_document(prayer))

    return ret_object


def judgments_data(query, page_number=1, page_size=10):
    ret_object = {
        "total_count": 0,
        "judgments": [],
    }
    judgments_collection = db['judgments']

    # Split the query into keywords
    keywords = query.split(' ')
    keywords = keywords
    regex_queries = [{"search_column": {"$regex": keyword, "$options": "i"}} for keyword in keywords]

    regex_queries.append({"search_column": {"$regex": "Telangana|Andhra|Hyderabad", "$options": "i"}})
    # MongoDB query with AND logic
    query_filter = {
        "$and": regex_queries
    }

    # Calculate total count of matching documents
    total_count = judgments_collection.count_documents(query_filter)
    ret_object["total_count"] = total_count

    # Apply pagination
    skip = (page_number - 1) * page_size
    prayers = judgments_collection.find(query_filter).skip(skip).limit(page_size)

    for prayer in prayers:
        ret_object['judgments'].append(serialize_document(prayer))

    return ret_object

