// Edit
function saveContent(id) {
    let revisedContent = $('#document-input').val();
    var form = document.getElementById("dataForm");
    var formData = new FormData();
    formData.append("editInput", true);
    formData.append("id", id);
    formData.append("revise_result", revisedContent);

    $('#loading-spinner').show();

    // Fetch API to submit the form data
    fetch(form.action, {
        method: "PUT",
        body: formData
    })
        .then(response => {
            if (!response.ok) {
                throw new Error("Network response was not ok");
            }
            console.log("editForm submitted successfully");
            $('#loading-spinner').hide();
            $("#editModal").modal("hide");
            window.location.replace(window.location.href);
        })
        .catch(error => {
            console.error("Error submitting form:", error.message);
            $('#loading-spinner').hide();
        });
}

// Delete
function confirmDelete() {
    let id = $('#deleteModal').attr('data-id');
    var form = document.getElementById("deleteForm");
    var formData = new FormData();
    formData.append("deleteInput", true);
    formData.append("id", id);

    $('#loading-spinner').show(); // 顯示 Loading 動畫

    fetch(
        form.action,
        {
            method: "DELETE",
            body: formData,
        })
        .then(response => {
            if (!response.ok) {
                throw new Error("Network response was not ok");
            }
            console.log("deleteForm submitted successfully");
            $('#loading-spinner').hide();
            $('#deleteModal').modal('hide');

            window.location.replace(window.location.href);
        })
        .catch(error => {
            console.error("Error submitting form:", error.message);
            $('#loading-spinner').hide();
        });

}


// Upload
function openFileSelection() {
    const fileInput = document.getElementById("fileInput");
    fileInput.accept = ".csv, .pdf, .txt, .doc, .docx";
    fileInput.addEventListener("change", handleFileSelection);
    fileInput.click();
}

function handleFileSelection(event) {
    const fileInput = document.getElementById("fileInput");
    fileInput.removeEventListener("change", handleFileSelection);
    // You can perform some checks on the files here if you want
    // Open the modal after file selection
    const uploadModal = new bootstrap.Modal(
        document.getElementById("uploadModal"),
    );
    uploadModal.show();
}

function submitForm(action) {
    var form = document.getElementById("uploadForm");
    var input = document.createElement("input");
    input.type = "hidden";
    input.name = "action";
    input.value = action;

    form.appendChild(input);

    if (action == "add" || action == "reset") {
        $("#uploadModal").modal("hide");
    }

    // After the form is submitted, close the current modal and open the new one.
    $("#uploadModal").on("hidden.bs.modal", function () {
        $("#ingesting-modal").modal("show");
    });

    form.submit();
    form.reset();
}