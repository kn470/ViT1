import torch
import torch.nn as nn
from data import quartered_dataset, train_loader
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import numpy as np
import torch.nn.functional as F


""" 
examples = iter(train_loader)
example_therm, example_label = next(examples) """

""" print(f"example_therm shape : {example_therm.shape}")
#so our input to the model is of size (50,256,320), because we set a batch size of 50
#now we should change the shape to (50, 8*8, 40*32)
#to represent (number of images in batch, number of patches altogether, number of pixels in each patch)
projection = nn.Sequential(
    Rearrange('b (h s1) (w s2) -> b (h w) (s1 s2)',h=8, w=8),
    #nn.Linear(32*40, 300)
)

print(f"projected tensor shape : {projection(example_therm).shape}") """

class ViT(nn.Module):

    def __init__(self, batch_size):
        super(ViT, self).__init__()
        self.n_patches = 8
        self.batch_size = batch_size
        self.hidden_dim = 300
        self.projection1 = nn.Linear(1280, 300)
        class_token = nn.Parameter(torch.rand(1,300))
        self.class_tokens = repeat(class_token, "b e -> n b e", n=batch_size)
        self.layer_norm = nn.LayerNorm(300)
        self.queries = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.keys = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.values = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.att_drop = nn.Dropout()
        self.MLP = nn.Sequential(
            nn.Linear(300, 1200),
            nn.GELU(),
            #include dropout here
            nn.Dropout(),
            nn.Linear(1200, 300)
        )
        self.classifier = nn.Linear(300,1)

    def patch(self, images):
        patched = rearrange(images, 'b (h s1) (w s2) -> b (h w) (s1 s2)',h=self.n_patches, w=self.n_patches)
        return patched
    
    def pos_embeds(self, token_shape):
        embeddings = torch.zeros(1,token_shape[0],token_shape[1])
        for i in range(token_shape[0]):
            for j in range(token_shape[1]):
                if j % 2 == 0:
                    embeddings[0,i,j] = np.sin(i/(10000**(j/token_shape[1])))
                else:
                    embeddings[0,i,j] = np.cos(i/(10000**((j-1)/token_shape[1])))

        pos_embed = nn.Parameter(embeddings)
        pos_embed.requires_grad = False
        return pos_embed

    def self_attention(self, input, n_heads):
        queries = rearrange(self.queries(input), 'b n (h d) -> b n h d', h=n_heads)
        keys = rearrange(self.keys(input), 'b n (h d) -> b n h d', h=n_heads)
        values = rearrange(self.values(input), 'b n (h d) -> b n h d', h=n_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        att = F.softmax(energy, dim=-1)/(self.hidden_dim ** 0.5)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, 'b n h d -> b n (h d)')
        return out





    def forward(self, images):
        #split up images into patches
        out = self.patch(images)
        #run linear projection
        out = self.projection1(out)
        #add the positional embeddings
        out = out + self.pos_embeds([out.shape[1], out.shape[2]])
        #add the cls tokens
        out = torch.cat([self.class_tokens, out], dim=1)
        #save copy of residual1
        resid1 = out.detach().clone()
        #apply layer normalisation
        out = self.layer_norm(out)
        #apply self attention once (for now)
        out = self.self_attention(out, n_heads=2)
        #add residual, and save second residual
        out = out + resid1
        resid2 = out.detach().clone()
        #apply second layer norm
        out = out = self.layer_norm(out)
        #apply MLP
        out = self.MLP(out)
        #apply classification to class token
        class_tokens = out[:,0]
        out = self.classifier(class_tokens)
        
        

        return out

    
        
        
model = ViT(batch_size=50)
lr = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#print(model(example_therm).shape)









