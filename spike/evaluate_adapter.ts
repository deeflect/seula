import { SieveModel, parseWithOperatorAliases } from '../src/engine';
import type { SchemaDef, FilterClause } from '../src/engine';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { filterDatasetRows } from '../src/datasets';

const schemas: Record<string, SchemaDef> = {
  catalog: { name:'catalog',description:'Synthetic catalog',split:'transfer',fields:[
    {name:'unit_price_usd',kind:'NUMBER',aliases:['price'],enumValues:[]},
    {name:'stock_qty',kind:'NUMBER',aliases:['stock'],enumValues:[]},
    {name:'rating',kind:'NUMBER',aliases:['stars'],enumValues:[]},
    {name:'label',kind:'TEXT',aliases:['name'],enumValues:[]},
    {name:'availability',kind:'ENUM',aliases:[],enumValues:['available','backorder','retired']},
  ]},
  devices: { name:'devices',description:'Synthetic devices',split:'transfer',fields:[
    {name:'draw_watts',kind:'NUMBER',aliases:['watts'],enumValues:[]},
    {name:'temp_c',kind:'NUMBER',aliases:['temperature'],enumValues:[]},
    {name:'error_count',kind:'NUMBER',aliases:['errors'],enumValues:[]},
    {name:'label',kind:'TEXT',aliases:['name'],enumValues:[]},
    {name:'mode',kind:'ENUM',aliases:[],enumValues:['online','standby','offline']},
  ]},
};
const rows: Record<string, Array<{id:string; fields:Record<string,string|number>}>> = {
 catalog:[
  {id:'C0',fields:{unit_price_usd:0,stock_qty:0,rating:0,label:'at least 10',availability:'available'}},
  {id:'C1',fields:{unit_price_usd:3,stock_qty:3,rating:3,label:'surpassing',availability:'backorder'}},
  {id:'C2',fields:{unit_price_usd:10,stock_qty:10,rating:10,label:'at most',availability:'retired'}},
  {id:'C3',fields:{unit_price_usd:21,stock_qty:12,rating:4.5,label:'no more than',availability:'available'}},
  {id:'C4',fields:{label:'beating 3'}},
 ],
 devices:[
  {id:'D0',fields:{draw_watts:0,temp_c:-10,error_count:0,label:'beneath',mode:'online'}},
  {id:'D1',fields:{draw_watts:3,temp_c:-3,error_count:3,label:'no fewer than',mode:'standby'}},
  {id:'D2',fields:{draw_watts:10,temp_c:0,error_count:10,label:'equaling',mode:'offline'}},
  {id:'D3',fields:{draw_watts:12,temp_c:3,error_count:1,label:'eclipsing 10',mode:'online'}},
  {id:'D4',fields:{temp_c:10,label:'at least 3'}},
 ],
};
const aliases = {
 topping:'gt', surpassing:'gt', eclipsing:'gt', beating:'gt', beneath:'lt', equaling:'eq',
 'at least':'gte', 'no fewer than':'gte', 'at most':'lte', 'no more than':'lte',
 'greater than':'gt', 'less than':'lt',
} as const;
const raw = `
A01|catalog|price topping 3|unit_price_usd:gt:3
A02|catalog|stock surpassing 0|stock_qty:gt:0
A03|catalog|stars eclipsing 3|rating:gt:3
A04|catalog|price beating 10|unit_price_usd:gt:10
A05|catalog|stock beneath 10|stock_qty:lt:10
A06|catalog|rating equaling 0|rating:eq:0
A07|catalog|stock   at least   3|stock_qty:gte:3
A08|catalog|stock no fewer than 10|stock_qty:gte:10
A09|catalog|price at most 3|unit_price_usd:lte:3
A10|catalog|rating no more than 3|rating:lte:3
A11|devices|watts greater than 3|draw_watts:gt:3
A12|devices|temperature less than 0|temp_c:lt:0
A13|devices|draw_watts topping 0|draw_watts:gt:0
A14|devices|errors surpassing 3|error_count:gt:3
A15|devices|temp_c eclipsing -3|temp_c:gt:-3
A16|devices|error_count beating 0|error_count:gt:0
A17|devices|temperature beneath 3|temp_c:lt:3
A18|devices|watts equaling 10|draw_watts:eq:10
A19|devices|errors at least 10|error_count:gte:10
A20|devices|temperature no fewer than -3|temp_c:gte:-3
A21|devices|watts at most 10|draw_watts:lte:10
A22|devices|errors no more than 3|error_count:lte:3
A23|catalog|TOPPING 3 PRICE|unit_price_usd:gt:3
A24|devices|No MoRe ThAn 10 WATTS|draw_watts:lte:10
G01|catalog|unit_price_usd > 3|unit_price_usd:gt:3
G02|catalog|stock_qty <= 10|stock_qty:lte:10
G03|catalog|rating != 3|rating:neq:3
G04|devices|draw_watts = 0|draw_watts:eq:0
G05|devices|temp_c < -3|temp_c:lt:-3
G06|devices|error_count >= 10|error_count:gte:10
G07|catalog|price > 0 AND stock < 10|unit_price_usd:gt:0&stock_qty:lt:10
G08|devices|watts <= 10 AND errors != 3|draw_watts:lte:10&error_count:neq:3
T01|catalog|label = 'surpassing'|label:eq:surpassing
T02|catalog|name contains 'at least'|label:contains:at least
T03|catalog|availability = available|availability:eq:available
T04|catalog|label = 'at most' AND price topping 0|label:eq:at most&unit_price_usd:gt:0
T05|devices|name contains 'no fewer than'|label:contains:no fewer than
T06|devices|mode = standby|mode:eq:standby
T07|devices|label = 'eclipsing 10' AND watts at most 10|label:eq:eclipsing 10&draw_watts:lte:10
T08|devices|name = 'BENEATH' AND temperature < 0|label:eq:beneath&temp_c:lt:0
R01|catalog|price hovering 3|REJECT
R02|catalog|stock outstripping 3|REJECT
R03|devices|watts around 10|REJECT
R04|devices|temperature warmer than 0|REJECT
R05|devices|errors approximately 3|REJECT
R06|catalog|price topping 3 OR stock beneath 10|REJECT
R07|catalog|(price topping 3)|REJECT
R08|catalog|price + stock topping 10|REJECT
R09|catalog|price topping stock|REJECT
R10|catalog|price topping 0 AND stock at least 0 AND rating beneath 10 AND price no more than 10 AND stock equaling 3 AND price topping 3|REJECT
R11|catalog|price topping|REJECT
R12|catalog|topping price|REJECT
R13|catalog|stock at least|REJECT
R14|catalog|price topping three|REJECT
R15|catalog|price topping 3 10|REJECT
R16|catalog|label topping 3|REJECT
R17|catalog|availability beneath 3|REJECT
R18|devices|mode at most 3|REJECT
R19|catalog|price topping '3'|REJECT
R20|devices|watts no more than '10'|REJECT
`;
const cases = raw.trim().split('\n').map(line=>{const [id,schema,query,expected]=line.split('|');return {id,schema,query,expected};});
if (cases.length !== 60 || new Set(cases.map(c=>c.id)).size !== 60) throw new Error('Expected 60 distinct cases');
const model = new SieveModel();
function wantClauses(encoded:string): FilterClause[] { return encoded.split('&').map(part=>{const [field,cmp,...value]=part.split(':');return {field,cmp:cmp as FilterClause['cmp'],value:value.join(':'),negated:false};}); }
function exact(a:readonly FilterClause[],b:readonly FilterClause[]):boolean {return JSON.stringify(a)===JSON.stringify(b);}
function semantic(clauses:readonly FilterClause[]):FilterClause[] {return clauses.map(c=>{const opposite=({eq:'neq',neq:'eq',gt:'lte',gte:'lt',lt:'gte',lte:'gt'} as Partial<Record<FilterClause['cmp'],FilterClause['cmp']>>)[c.cmp];return c.negated&&opposite?{...c,cmp:opposite,negated:false}:c;});}
function matches(row:Record<string,string|number>,clauses:readonly FilterClause[]):boolean {
 return clauses.every(c=>{const v=row[c.field]; if(v===undefined) return false; const a=typeof v==='number'?v:String(v).toLowerCase();const b=typeof v==='number'?Number(c.value):c.value.toLowerCase();let ok=false;
  switch(c.cmp){case'eq':ok=a===b;break;case'neq':ok=a!==b;break;case'gt':ok=typeof a==='number'&&a>Number(b);break;case'gte':ok=typeof a==='number'&&a>=Number(b);break;case'lt':ok=typeof a==='number'&&a<Number(b);break;case'lte':ok=typeof a==='number'&&a<=Number(b);break;case'contains':ok=typeof a==='string'&&a.includes(String(b));break;} return c.negated?!ok:ok;
 });
}
function ids(schema:string,clauses:readonly FilterClause[]):string[]{return rows[schema].filter(r=>matches(r.fields,clauses)).map(r=>r.id);}
function actualIds(schema:string,clauses:readonly FilterClause[]):string[]{return filterDatasetRows(rows[schema].map(r=>({...r,title:r.id,subtitle:'',badges:[],rightMeta:'',rightSub:''})),[...clauses]).map(r=>r.id);}
const results = cases.map(c=>{
 const schema=schemas[c.schema]; const adapted=parseWithOperatorAliases(model,c.query,schema,aliases); const baseline=model.parse(c.query,schema); const expectedReject=c.expected==='REJECT'; const wanted=expectedReject?[]:wantClauses(c.expected);const oracleRows=expectedReject?[]:ids(c.schema,wanted);
 const report=(res:typeof baseline)=>({accepted:res.accepted,clauses:res.clauses,issues:res.issues,exact:!expectedReject&&res.accepted&&exact(res.clauses,wanted),semanticExact:!expectedReject&&res.accepted&&exact(semantic(res.clauses),semantic(wanted)),rows:res.accepted?actualIds(c.schema,res.clauses):[],rowExact:!expectedReject&&res.accepted&&JSON.stringify(actualIds(c.schema,res.clauses))===JSON.stringify(oracleRows),safe:expectedReject&&!res.accepted});
 return {...c,expectedReject,wanted,oracleRows,normalizedQuery:adapted.normalizedQuery,rewrites:adapted.rewrites,adapter:report(adapted.result),baseline:report(baseline)};
});
function summary(key:'adapter'|'baseline') { const valid=results.filter(r=>!r.expectedReject), rejects=results.filter(r=>r.expectedReject);return {validAccepted:valid.filter(r=>r[key].accepted).length,validExact:valid.filter(r=>r[key].exact).length,validSemanticExact:valid.filter(r=>r[key].semanticExact).length,validRowExact:valid.filter(r=>r[key].rowExact).length,validWrong:valid.filter(r=>r[key].accepted&&!r[key].semanticExact).length,validRejected:valid.filter(r=>!r[key].accepted).length,rejectSafe:rejects.filter(r=>r[key].safe).length,rejectAccepted:rejects.filter(r=>r[key].accepted).length};}
const output={stage:'development recheck after missing operator-value-field order',sourceSha:createHash('sha256').update(readFileSync('src/engine/vocabulary.ts')).digest('hex'),cases:cases.length,adapter:summary('adapter'),baseline:summary('baseline'),misses:results.filter(r=>r.expectedReject?!r.adapter.safe:!r.adapter.semanticExact).map(r=>({id:r.id,query:r.query,expected:r.expected,oracleRows:r.oracleRows,normalizedQuery:r.normalizedQuery,adapted:r.adapter,baseline:r.baseline}))};
writeFileSync('docs/adapter-development-results.jsonl',results.map(r=>JSON.stringify(r)).join('\n')+'\n');
writeFileSync('docs/adapter-development-summary.json',JSON.stringify(output,null,2)+'\n');
console.log(JSON.stringify({adapter:output.adapter,baseline:output.baseline,misses:output.misses.map(m=>m.id)},null,2));
