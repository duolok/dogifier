package main

import (
	"fmt"
    "io"
    "log"
    "net/http"
    "os"
    "path/filepath"
    "github.com/a-h/templ"
    "github.com/a-h/templ/loader"
    "github.com/a-h/templ/render"
)

func main() {
	http.HandleFunc("/", serveTemplate)
	http.HandleFunc("/upload", uploadFile)
	http.Handle("/static", http.StripPrefix("/static/"), http.FIleServer(http.Dir("static")))

	fmt.Println("Server started on :8080")
	log.Fatalf(http.ListenAndServe(":8080", nil))
}

func serveTemplate(writer http.ResponseWriter, r *http.Request) {
	tmpl, err := loader.Load("index.templ")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	err = render.HTML(writer, tmpl, nil)
	if err != nil {
		http.Error(writer, err.Error(), http.StatusInternalServerError)
	}
}

func uploadFile(writer http.ResponseWriter, request *http.Request) {
	if request.Method  != "POST" {
		http.Error(writer, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(writer, err.Error(), http.StatusInternalServerError)
	}
	defer file.Close()

	out, err := os.Create(filepath.Join("static/uploads", header.Filename))
	if err != nil {
		http.Error(writer, err.Error(), http.StatusInternalServerError)
		return
	}
	defer out.Close()
	
	_, err = io.Copy(out, file)
	if err != nil {
		http.Error(writer, err.Error(), http.StatusInternalServerError)
		return
	}

	fmt.Fprintf(writer, "File uploaded successfully: %s\n", header.Filename)
}
