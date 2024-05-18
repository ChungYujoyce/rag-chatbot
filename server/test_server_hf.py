import transformers
import torch

model_path = "/data/nlp/meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "As a helpful assistant, you will utilize the provided document to answer user questions. \nRead the given document before providing answers and think step by step.\nThe document has an order of paragraphs with a higher correlation to the questions from the top to the bottom. \nThe answer may be hidden in the tables, so please find it as closely as possible. \nDo not use any other information to answer the user. Provide a detailed answer to the question.\nAlso, please provide the answer in the following order of priorities if applicable:\nFirstly, emphasize GPU characteristics and GPU products.\nSecondly, Give prominence to power-related specifications such as fan cooling or liquid cooling, power consumption, and so on.\nThirdly, If applicable, mention green computing.\nRemember, please don't provide any fabricated information, ensuring that everything stated is accurate and true."},
    {"role": "user", "content": "The followings are the retrieved documents: SYS-421GE-TNRT3\nDATASHEET\nGPU SuperServer SYS-421GE-TNRT3\n4U Dual Processor (5th/4th Gen Intel Xeon) System with Support for 8 PCIe based GPUs\nMore details here\nKey Applications\nHigh Performance Computing, VDI, AI/Deep Learning Training, Media/Video\nStreaming, Cloud Gaming, Animation and Modeling, Design & Visualization, 3D\nRendering, Diagnostic Imaging,\nKey Features\n5th/4th Gen Intel Xeon Scalable processor support;\n32 DIMM slots Up to 8TB: 32x 256 GB DRAM Memory Type: 5600MTs ECC\nDDR5;\n8 PCIe Gen 5.0 X16 FHFL Slots; AIOM/OCP 3.0 Support;\n8x 2.5\" Hot-swap SATA drive bays; 8 HOT SWAP 2.5 SATA/SAS (AOC\nrequired); 4x2.5\" Hot-swap NVMe drive bays;\n8 Hot-Swap Heavy Duty Fans with Optimal Fan Speed Control;\n4x 2700W (2+2) Redundant Power Supplies, Titanium Level;\n\nForm Factor,\"4U Rackmount Enclosure: 437X 178 x 737mm (17.2\"\"x7\"\" x29\"\") Package: (27\"\" x 26.57\"\" x 41\"\")\"\nProcessor,Dual Socket E (LGA-4677) Sth Gen Intele Xeone/4th Gen Intele Xeone Scalable processors Up to 64C/128T; Up to 320MB Cache per CPU\nGPU,\"Max GPU Count: Up to 8 double-width or 8 single-width GPU(s) Supported GPU: NVIDIA PCle: H1OO,RTX A4000,L4OS,ALOO,RTX 6000 ADA CPU-GPU Interconnect: PCle 5.0 xl6 CPU-to-GPU Interconnect GPU-GPU Interconnect: NVIDIA@ NVLinke Bridge (optional)\"\nSystem Memory,Slot Count: 32 DIMM slots Max Memory (1DPC): Up to 4TB 5600MT/s ECC DDRS RDIMM Max Memory (2DPC): Up to 8TB 4400MT/s ECC DDRS RDIMM\nDrive Bays,\"Max Memtoly (ZuFe: Up [u 0TD #4uuivi/> ECC DDn) nDIMM 24x 2.5\"\" hot-swap NVMe/SATA/SAS drive bays (4x 2.5\"\" NVMe hybrid; 4x 2.5\"\" NVMe dedicated;\"\nExpansion Slots,12 PCle 5.0 x16 FHFL slot(s)\nOn-Board Devices,Chipset: Intele C741 Network Connectivity: 2x 1OGbE BaseT with Intele X710-AT2 IPMI: Support for Intelligent Platform Management Interface v.2.0\nInput Output,Cpr Video: 1 VGA port(s)\n\nARS-111GL-NHR\nDATASHEET\nGPU SuperServer ARS-111GL-NHR\n1U NVIDIA Grace-Hopper Super Chip with On board Hopper GPU and 72 core Grace CPU\n\n\nQuestion: how are you?"},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=512,
    eos_token_id=terminators,
    do_sample=False,
)
print(outputs[0]["generated_text"][len(prompt):])