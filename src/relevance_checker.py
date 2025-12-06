def is_skin_related(query: str) -> bool:
    keywords = ["skin","pimple","acne","eczema","rash","hair","scalp","dandruff","psoriasis","derma"]
    return any(k in query.lower() for k in keywords)
