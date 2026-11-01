const promptBox=document.querySelector("#prompt");
const count=document.querySelector("#prompt-count");
const generateButton=document.querySelector("#generate");
const downloadButton=document.querySelector("#download");
const message=document.querySelector("#builder-message");
const preview=document.querySelector("#preview");
const sourceCode=document.querySelector("#source-code");
let latestFiles=null;
promptBox.addEventListener("input",()=>count.textContent=promptBox.value.length+" / 1200");
document.querySelectorAll(".tab").forEach(button=>button.addEventListener("click",()=>{
  document.querySelectorAll(".tab,.panel").forEach(item=>item.classList.remove("active"));
  button.classList.add("active");
  document.querySelector("#"+button.dataset.panel).classList.add("active");
}));
async function checkHealth(){
  const label=document.querySelector("#system-status");
  const dot=document.querySelector(".status-dot");
  try{
    const response=await fetch("/api/health");
    const data=await response.json();
    if(!response.ok)throw new Error("API unavailable");
    label.textContent=data.status==="ok"?"Python API online":"API unavailable";
    dot.classList.toggle("ok",data.status==="ok");
  }catch(error){label.textContent="Preview mode";dot.classList.remove("ok");}
}
generateButton.addEventListener("click",async()=>{
  const prompt=promptBox.value.trim();
  if(!prompt){message.textContent="Please describe the website first.";message.className="message error";return;}
  generateButton.disabled=true;generateButton.textContent="Generating…";
  message.textContent="Python is generating your starter website…";message.className="message";
  downloadButton.disabled=true;
  try{
    const response=await fetch("/api/generate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({prompt})});
    const data=await response.json();
    if(!response.ok)throw new Error(data.error||"Website generation failed.");
    latestFiles=data.files;
    const documentHtml=data.files["index.html"].replace('<link rel="stylesheet" href="styles.css">',"<style>"+data.files["styles.css"]+"</style>").replace('<script src="script.js"></script>',"<script>"+data.files["script.js"]+"<"+"/script>");
    preview.srcdoc=documentHtml;
    sourceCode.textContent=data.files["index.html"];
    downloadButton.disabled=false;
    message.textContent="Generated "+data.kind+" starter website. Preview is isolated in a sandboxed iframe.";
  }catch(error){message.textContent=error.message;message.className="message error";}
  finally{generateButton.disabled=false;generateButton.textContent="Generate preview ↗";}
});
downloadButton.addEventListener("click",async()=>{
  if(!latestFiles)return;
  message.textContent="Preparing ZIP download…";
  try{
    const response=await fetch("/api/generate?format=zip",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({prompt:promptBox.value.trim()})});
    if(!response.ok)throw new Error("ZIP export unavailable.");
    const blob=await response.blob();const url=URL.createObjectURL(blob);
    const link=document.createElement("a");link.href=url;link.download="pymind-website.zip";link.click();URL.revokeObjectURL(url);
    message.textContent="ZIP download started.";
  }catch(error){message.textContent=error.message;message.className="message error";}
});
checkHealth();