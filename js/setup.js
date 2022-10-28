function(width, height, size){
    let app=document.querySelector("gradio-app");
    app=app.shadowRoot??app;
    app.querySelector("#sdinfframe").style.height=80+Number(height)+"px";
    app.querySelector("#setup_row").style.display="none";
    let frame=app.querySelector("#sdinfframe").contentWindow.document;

    if(frame.querySelector("#setup").value=="0")
    {
        window.my_setup=setInterval(function(){
            let app=document.querySelector("gradio-app");
            app=app.shadowRoot??app;
            let frame=app.querySelector("#sdinfframe").contentWindow.document;
            console.log("Check PyScript...")
            if(frame.querySelector("#setup").value=="1")
            {
                frame.querySelector("#draw").click();
                clearInterval(window.my_setup);
            }
        }, 100)
    }
    else
    {
        frame.querySelector("#draw").click();
    }
    return [width, height, size];
}