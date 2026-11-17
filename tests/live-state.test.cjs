// Value: protects matching configurations across async UI results; seam=none.
// Full app/controller, fake DOM and controlled transport. Real browser/OpenMC
// evidence is separate; these checks do not validate transport physics.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function harness() {
  const elements = new Map(), calls = [];
  class Element {
    constructor(){this.value='';this.checked=false;this.disabled=false;this.hidden=false;this.dataset={};this.handlers={};this.classList={toggle(){}};}
    addEventListener(type,fn){this.handlers[type]=fn;}
    setAttribute(){}
    getContext(){return null;}
    getBoundingClientRect(){return {width:400,height:200};}
  }
  const el=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
  let responder;
  const context=vm.createContext({
    document:{getElementById:el,querySelectorAll:()=>[],querySelector:()=>new Element()},
    window:{devicePixelRatio:1,addEventListener(){}},ResizeObserver:class{observe(){}},
    structuredClone,console,setTimeout,clearTimeout,TextEncoder,TextDecoder,Uint8Array,Map,Promise,URL,Blob,
    fetch:async(url,options)=>{
      if(url==='atlas.json')return new Promise(()=>{});
      calls.push({url,payload:JSON.parse(options.body)});
      return {ok:true,json:async()=>responder(url,JSON.parse(options.body))};
    },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8'),context);
  vm.runInContext(`
    // Rendering is outside these controller checks and is covered in the browser.
    renderFields=()=>{};updateActions=()=>{};drawGeometry=()=>{};
    state.api.prediction=true;state.api.solver=true;
    state.atlas={cases:[3,5].map(thickness=>({
      id:'archive-'+thickness,split:'test',family:'solid',
      config:{family:'solid',thickness,density_scale:1,source_y:0,fast_fraction:.8},
      prediction:[[thickness],[thickness]],uncertainty:[[.1],[.1]],
      reference:[[thickness],[thickness]],stderr:[[.01],[.01]],metrics:{}
    }))};
    loadCase(0);
  `,context);
  return {context,el,calls,run:code=>vm.runInContext(code,context),respond:fn=>{responder=fn;}};
}

function deferred(){let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};}
const model=config=>({prediction:[[config.thickness],[config.thickness]],uncertainty:[[.1],[.1]],seconds:.01,ood:false,config});
const tick=()=>new Promise(r=>setImmediate(r));
function change(h,value){h.el('thickness').value=value;h.el('thickness').handlers.input();}

test('changed draft remains dirty and a later reference obtains its matching prediction',async()=>{
  const h=harness(),pending=deferred();
  h.respond((url,body)=>url==='api/predict'?(body.thickness===3?pending.promise:model(body)):
    {reference:[[body.config.thickness],[body.config.thickness]],stderr:[[.01],[.01]],seconds:.1,seed:body.seed,histories:40000,config:body.config});
  const query=h.run('predict()');await tick();change(h,5);
  pending.resolve(model({family:'solid',thickness:3,density_scale:1,source_y:0,fast_fraction:.8}));
  await query;
  assert.equal(h.run('state.current.config.thickness'),3);
  assert.equal(h.run('state.dirty'),true);
  assert.equal(Number(h.el('thickness').value),5);
  await h.run('simulate()');
  assert.equal(h.calls.filter(call=>call.url==='api/predict').length,2);
  assert.equal(h.run('state.current.config.thickness'),5);
  assert.equal(h.run('state.current.prediction[0][0]'),5);
  assert.equal(h.run('state.current.reference[0][0]'),5);
  assert.equal(h.run('state.dirty'),false);
});

test('late prediction cannot replace a newer archive selection',async()=>{
  const h=harness(),pending=deferred();h.respond(()=>pending.promise);
  const query=h.run('predict()');await tick();h.run('loadCase(1)');
  pending.resolve(model({family:'solid',thickness:3,density_scale:1,source_y:0,fast_fraction:.8}));await query;
  assert.equal(h.run('state.current.id'),'archive-5');
  assert.equal(h.run('state.current.config.thickness'),5);
  assert.equal(h.run('state.busy'),false);
});

test('late reference preserves a newer draft and a newer case selection',async()=>{
  for(const select of [false,true]){
    const h=harness(),pending=deferred();h.respond(()=>pending.promise);
    const compute=h.run('simulate()');await tick();
    if(select)h.run('loadCase(1)');else change(h,5);
    pending.resolve({reference:[[3],[3]],stderr:[[.01],[.01]],seconds:.1,histories:40000});await compute;
    assert.equal(h.run('state.current.config.thickness'),select?5:3);
    assert.equal(h.run('state.dirty'),!select);
    assert.equal(h.run('state.current.prediction[0][0]'),select?5:3);
    assert.equal(h.run('state.current.reference[0][0]'),select?5:3);
    assert.equal(h.run('state.busy'),false);
  }
});
