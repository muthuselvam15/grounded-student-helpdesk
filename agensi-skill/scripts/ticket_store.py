import argparse, json, sys, uuid
from datetime import datetime, timezone
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description="Create/list local JSON helpdesk tickets.")
    p.add_argument("--file",default="data/tickets.json")
    s=p.add_subparsers(dest="command",required=True)
    c=s.add_parser("create"); c.add_argument("--question",required=True); c.add_argument("--reason",required=True)
    s.add_parser("list")
    a=p.parse_args(); f=Path(a.file)
    try:
        rows=json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
        if not isinstance(rows,list): raise ValueError("Ticket file must contain a JSON array.")
        if a.command=="create":
            q,r=a.question.strip(),a.reason.strip()
            if not q or not r: raise ValueError("Question and reason must not be empty.")
            t={"id":str(uuid.uuid4()),"question":q,"reason":r,
               "created_at":datetime.now(timezone.utc).isoformat(),"status":"open"}
            rows.append(t); f.parent.mkdir(parents=True,exist_ok=True)
            tmp=f.with_suffix(f.suffix+".tmp")
            tmp.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); tmp.replace(f)
            print(json.dumps({"ok":True,"message":"Ticket saved locally.","ticket":t},indent=2))
        else: print(json.dumps({"ok":True,"count":len(rows),"tickets":rows},ensure_ascii=False,indent=2))
        return 0
    except Exception as e:
        print(json.dumps({"ok":False,"error":str(e)},indent=2)); return 1
if __name__=="__main__": sys.exit(main())
