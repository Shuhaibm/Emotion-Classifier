function onSubmit() {
    link = "emotion-classifier.shuhaibmehri.repl.co/api?string=" + document.getElementById("inputString").value;
    console.log(link);
    window.open(link);
}