highlightText(document.body)function highlightText(element){    if (element.hasChildNodes()){        element.childNodes.forEach(highlightText)    } else if (element.nodeType === Text.TEXT_NODE){        console.log("Yuup")        console.log(String(element.innerText))     //   element.textContent = element.textContent.replace(/the/gi,"||||")            }}