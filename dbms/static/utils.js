function isTableLine(line) {
    // Check if the line contains the pipe character '|'
    return line.includes('|');
}

function displayTextInTable(button, text) {
    
    var parentDiv = button.parentNode.parentNode;
    var contentDiv = parentDiv.querySelector("span");

    const lines = text.split('\n');
    let inTable = false; // flag to check if we're currently in the table
    if (contentDiv.children.length === 0) {
        contentDiv.innerHTML = ''; // Clear existing content
        lines.forEach((line, index) => {
            if (isTableLine(line)) {
                if (!inTable) {
                    inTable = true; // we're entering the table
                    const table = document.createElement('table');
                    table.className = 'table table-bordered'; // Bootstrap table style
                    const row = document.createElement(index === 0 ? 'thead' : 'tr');
                    const cells = line.split('|').map(cell => cell.trim());
                    cells.forEach(cell => {
                        const element = index === 0 ? 'th' : 'td';
                        const td = document.createElement(element);
                        td.textContent = cell;
                        row.appendChild(td);
                    });
                    table.appendChild(row);
                    contentDiv.appendChild(table);
                } else {
                    const row = document.createElement('tr');
                    const cells = line.split('|').map(cell => cell.trim());
                    cells.forEach(cell => {
                        const td = document.createElement('td');
                        td.textContent = cell;
                        row.appendChild(td);
                    });
                    contentDiv.lastChild.appendChild(row);
                }
            } else {
                if (inTable) {
                    inTable = false; // we're exiting the table
                }
                const div = document.createElement('div');
                div.textContent = line;
                contentDiv.appendChild(div);
            }
        });
    };
}

function getRevisedContent(button) {
        
    var parentDiv = button.parentNode;
    var contentDiv = parentDiv.querySelector("span");
    let revisedContent = ''; // Initialize an empty string to store the revised content

    // Loop through all child nodes of the contentDiv
    contentDiv.childNodes.forEach(node => {
        // Check if the node is a table
        if (node.tagName && node.tagName.toLowerCase() === 'table') {
            // Add a newline character above the table
            revisedContent += '\n\n';

            // Loop through the rows of the table
            node.querySelectorAll('tr').forEach(row => {
                // Loop through the cells of each row
                row.querySelectorAll('td, th').forEach((cell, index) => {

                    // Add the cell content to the revised content string
                    revisedContent += cell.innerText.replace(/\s+/g, " ").trim();
                    // Add "|" if it's not the last cell in the row
                    if (index < row.cells.length - 1) {
                        revisedContent += ' | ';
                    }
                });

                // Add a newline character after each row
                revisedContent += '\n';
            });

            // Add a newline character below the table
            revisedContent += '\n\n';
        } else {
            revisedContent += node.innerText.replace(/\s+/g, " ").trim();
            revisedContent += '\n';
        }
    });
    revisedContent = revisedContent.replace(/\n{3,}/g, "\n\n");
    revisedContent = revisedContent.trim();
    // console.log("Revised Content:", revisedContent); // Print the revised content to the console
    alert("Revised Content:\n" + revisedContent); // Display the revised content in an alert dialog
    return revisedContent; // Return the revised content
}

function toggleEditing(button) {
    // $('#confirmationModal').modal('show');
    // $('#confirmAction').on('click', function() {
        
    var parentDiv = button.parentNode;
    var contentDiv = parentDiv.querySelector("span");
    const reviseButton = button;
    const saveButton = parentDiv.querySelector(".saveButton");
    const isEditable = contentDiv.getAttribute('contenteditable') === 'true';
    console.log(isEditable);
    if (!isEditable) {
        originalContent = contentDiv.innerHTML; // Store the original content
        contentDiv.setAttribute('contenteditable', 'true'); // Make content editable
        contentDiv.style.display = "block";
        contentDiv.focus(); // Put focus on the contentDiv for immediate editing
        reviseButton.textContent = 'Cancel';
        saveButton.style.display = 'inline-block';
  
    } else {
        contentDiv.innerHTML = originalContent; // Reset content to original state
        contentDiv.removeAttribute('contenteditable'); // Make content non-editable
        reviseButton.textContent = 'Revise';
        saveButton.style.display = 'none';
    }
  // });
}

function saveContent(button, id) {
    const revisedContent = getRevisedContent(button);
    var parentDiv = button.parentNode;
    var contentDiv = parentDiv.querySelector("span");
    contentDiv.removeAttribute('contenteditable'); // Make content non-editable
    parentDiv.querySelector(".reviseButton").textContent = 'Revise';
    button.style.display = 'none';
    
    // id 
    // editedString
    var form = document.getElementById("dataForm");
    var formData = new FormData();
    formData.append("editInput", true);
    formData.append("id", id);
    formData.append("revise_result", revisedContent);

    // Fetch API to submit the form data
    fetch(form.action, {
        method: "PUT",
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Network response was not ok");
        }
        // Handle success response if needed
        console.log("Form submitted successfully");
    })
    .catch(error => {
        // Handle error if needed
        console.error("Error submitting form:", error.message);
    });
}

function deleteString(button, id, url) {
    event.preventDefault();

    $('#confirmationModal').modal('show');
    $('#confirmAction').off('click');
    $('#confirmAction').on('click', function() {
        var listItem = button.parentNode.parentNode;
        listItem.remove();
        
        var formData = new FormData();
        // id 
        // editedString
        formData.append("deleteInput", true);
        formData.append("id", id);

        // Fetch API to submit the form data
        fetch(url, {
            method: "DELETE",
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error("Network response was not ok");
            }
            // Handle success response if needed
            console.log("Form submitted successfully");
        })
        .catch(error => {
            // Handle error if needed
            console.error("Error submitting form:", error.message);
        });
    // Close the modal
    $('#confirmationModal').modal('hide');
  });
}

function openFileSelection() {
  const fileInput = document.getElementById("fileInput");
  fileInput.accept = ".csv, .pdf, .txt, .doc, .docx";
  fileInput.click();
  fileInput.addEventListener("change", handleFileSelection);
}

function handleFileSelection(event) {
  // You can perform some checks on the files here if you want
  // Open the modal after file selection
  const uploadModal = new bootstrap.Modal(
    document.getElementById("uploadModal"),
  );
  uploadModal.show();
}

function submitPromptForm() {
  // Show the modal
  $("#responseModal").modal("show");

  // Submit the form after a short delay to allow the modal to open
  setTimeout(function () {
    document.getElementById("promptForm").submit();
  }, 5);
}

function submitForm(action) {
  var form = document.getElementById("uploadForm");

  var input = document.createElement("input");
  input.type = "hidden";
  input.name = "action";
  input.value = action;

  form.appendChild(input);

  // After the form is submitted, close the current modal and open the new one.
  $("#uploadModal").on("hidden.bs.modal", function () {
    $("#ingesting-modal").modal("show");
  });

  if (action == "add" || action == "reset") {
    $("#uploadModal").modal("hide");
  }

  form.submit();
}
