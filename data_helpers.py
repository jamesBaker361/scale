from torch.utils.data import Dataset
from datasets import load_dataset
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTokenizer

class AnimalData(Dataset):
    def __init__(self,image_processor:VaeImageProcessor,
                 tokenizer:CLIPTokenizer,
                 dim:int=128,
                 hf_path:str="Rapidata/Animals-10"):
        super().__init__()
        self.data=load_dataset(hf_path,split="train")
        self.image_processor=image_processor
        self.tokenizer=tokenizer
        self.dim=dim
        self.mapping=['Butterfly', 'Cat', 'Chicken', 'Cow', 'Dog', 'Elephant', 'Horse', 'Sheep', 'Spider', 'Squirrel' ]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image=self.image_processor.preprocess(self.data["image"][index],self.dim,self.dim)[0]
        text=self.tokenizer(
            self.mapping[self.data[index]["label"]], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "image":image,
            "text":text,
            "text_str":self.data["label"][index],
        }
        