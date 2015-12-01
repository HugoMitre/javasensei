/* global usuario */

var ultimoMensaje;
var chatSensei = function () {
    this.idInterval = -1;
    Offline.options = {checks: {xhr: {url: '/favicon.ico'}}};
    this.verificarMensaje = function () {        
        //Ajax para saber si hay nuevos mensajes, verifica si hay conectividad
        Offline.check();
    };
    
    this.changeExercise = function(estado){
        var chatBoton = $("#chatboton");
        if (estado && ultimoMensaje){ //Se habilita y ademas se cambia el id
            chatBoton.removeClass("ui-state-disabled");
            this.idInterval = setInterval(this.verificarMensaje, 500);
        }else{
            chatBoton.addClass("ui-state-disabled");
            clearInterval(this.idInterval);
        }
        $("#chatbox").empty();
    };

    this.setUp = function () {
        $("#usermsg").keyup(this.procesarEnvioMensaje);
        Offline.on('up', function() {
            $.ajax({
                type: "GET",
                url: url + "chat/leermensajes",
                data: {
                    fechaActual: ultimoMensaje,
                    idEjercicio: avatar_context.id
                }
            }).done(function(datos){
                /*data.message = $("#usermsg").val()
                        .replace(/<br\/>/g,"\n")
                        .replace(/</g, "&lt;")
                        .replace(/>/g, "&gt;")
                        .replace(/\n/g, "<br/>");*/
                if (datos.length>0){
                    //Agregamos nuevos mensajes
                    $.each(datos,function(index,data){
                        $("#chatbox").append(
                                $("<p class='mensaje'>").html(data.nombreUsuario + ": " + datos.message)
                                .css("color",data.color));
                        var chatbox = document.getElementById("chatbox");
                        chatbox.scrollTop = chatbox.scrollHeight;
                    });

                    ultimoMensaje = datos[datos.length-1].fecha;
                }    
            });
        });
        //Obtenemos la hora del servidor
        obtenerHora();
    };

    this.procesarEnvioMensaje = function (e) {
        var code = (e.keyCode ? e.keyCode : e.which);
        var message = ("<br>"+$("#usermsg").val())
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/\n/g, "<br/>");
        var lastChar = message.length - 1;
        if(e.shiftKey && e.keyCode === 13) {
            //Espacio vacío
        } else if (code === 13) {
            var regex = /[a-z]|[0-9]|[áéíóúñ]|[$-/:-?{-~!"^_`\[\]]/i;
            if (!regex.test(message)){
                $("#usermsg").val("");
                return;
            }
            message = message.substring(0,lastChar);
            var color = $("#botoncolor").css("background-color");
            $("#usermsg").val("");
            $("#chatbox").append(
                    $("<p class='mensaje'>").html(usuario.nombre + ": " + message)
                    .css("color",color));
            chatbox = document.getElementById("chatbox");
            chatbox.scrollTop = chatbox.scrollHeight;
            //Enviar mensaje al servidor
            $.ajax({
                type: "GET",
                url: url + "chat/agregarmensaje",
                data: {
                    message: message,
                    nombreUsuario: usuario.nombre,
                    idUsuario: usuario.id,
                    idEjercicio: avatar_context.id,
                    color: color
                }
            }).done(function(data){
                if (data>ultimoMensaje)
                    ultimoMensaje = data;
            });
        }
    };

};

function obtenerHora(){
    $.get(url+"servidor/obtenerHora")
    .done(function(data){
        ultimoMensaje = data;
    }).fail(function(){
        setTimeout(1000, obtenerHora);
    });
}