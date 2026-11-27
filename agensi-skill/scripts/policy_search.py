import argparse, json, re, sys
from pathlib import Path

STOP = {"a","an","and","are","as","at","be","by","for","from","how","in","is","it","of","on","or","the","to","what","when","where","which","with","your"}
def terms(s):
    return {x for x in re.findall(r"[a-z0-9]+", str(s).lower()) if len(x)>1 and x not in STOP}

def main():
    p=argparse.ArgumentParser(description="Search approved university policy JSON.")
    p.add_argument("--query", required=True)
    p.add_argument("--policies", default="data/policies.json")
    p.add_argument("--limit", type=int, default=5)
    a=p.parse_args()
    f=Path(a.policies)
    if not f.exists():
        print(json.dumps({"ok":False,"error":f"Policy file not found: {f}","matches":[]},indent=2)); return 2
    try: rows=json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:
        print(json.dumps({"ok":False,"error":f"Cannot read policy JSON: {e}","matches":[]},indent=2)); return 2
    if not isinstance(rows,list):
        print(json.dumps({"ok":False,"error":"Policy file must be a JSON array.","matches":[]},indent=2)); return 2
    q=terms(a.query)
    found=[]
    for i,r in enumerate(rows):
        if not isinstance(r,dict) or not isinstance(r.get("title"),str) or not isinstance(r.get("text"),str): continue
        title, body=terms(r["title"]), terms(r["text"])
        overlap=q & (title|body)
        if not overlap: continue
        score=min(1.0, len(overlap)/max(1,len(q)) + .25*len(q&title)/max(1,len(q)))
        found.append({"score":round(score,4),"matched_terms":sorted(overlap),"policy":{
            "id":r.get("id",f"record-{i+1}"),"title":r["title"],"text":r["text"],
            "source":r.get("source"),"effective_date":r.get("effective_date")}})
    found.sort(key=lambda x:x["score"],reverse=True)
    found=found[:max(1,min(a.limit,20))]
    print(json.dumps({"ok":True,"query":a.query,"match_count":len(found),
      "warning":"Lexical score is only a ranking aid; inspect policy authority and evidence.",
      "matches":found},ensure_ascii=False,indent=2))
    return 0
if __name__=="__main__": sys.exit(main())
