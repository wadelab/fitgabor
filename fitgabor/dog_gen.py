import torch
from torch import nn

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

class DOGGenerator(nn.Module):
    #Based on the Gabor code but returns a difference of Gaussians: 
    def __init__(self, image_size, target_std=1.):
        """
        DOG generator class.
        Params:
            sigma1(float): std deviation of the center Gaussian.
            sigma2(float): std of the surround Gaussian
            amp1(float): amplitude of the centre
            amp2(float): amp of the surround
            center (tuple of integers): The position of the filter.
            image_size (tuple of integers): Image height and width.
            target_std:
        Returns:
            2D torch.tensor: A DOG filter.
        """
        
        super().__init__()
        self.sigma1 = nn.Parameter(torch.rand(1)*2, requires_grad=False)
        self.sigma2 = nn.Parameter(torch.rand(1)+2+1, requires_grad=False)
        self.amp1 = nn.Parameter(torch.zeros(1)+1., requires_grad=False)
        self.amp2 = nn.Parameter(torch.zeros(1)+1., requires_grad=False)

        self.center = nn.Parameter(torch.tensor([0., 0.]))
        self.image_size = image_size
        self.target_std = target_std
    
    def forward(self):
        return self.gen_dog()
    
    def gen_dog(self):
        
        # clip values in reasonable range
        self.theta.data.clamp_(-torch.pi, torch.pi)
        self.sigma1.data.clamp_(2., min(self.image_size)/2) #min(self.image_size)/7, min(self.image_size)/5) #2)
        self.sigma2.data.clamp_(2., min(self.image_size)/2) #min(self.image_size)/7, min(self.image_size)/5) #2)

        ymax, xmax = self.image_size
        xmax, ymax = (xmax - 1)/2, (ymax - 1)/2
        xmin = -xmax
        ymin = -ymax
        (y, x) = torch.meshgrid(torch.arange(ymin, ymax+1), torch.arange(xmin, xmax+1))
        dog = torch.exp(-.5 * (x ** 2 / sigma1 ** 2 + y ** 2 / sigma1 ** 2)) - torch.exp(-.5 * (x ** 2 / sigma2 ** 2 + y ** 2 / sigma ** 2))

        return dog.view(1, 1, *self.image_size)
    
    def apply_changes(self):
        self.sigma1.requires_grad_(True)
        self.sigma2.requires_grad_(True)



class DOGGenerator(nn.Module):
    #Based on the Gabor code but returns a difference of Gaussians: 
    def __init__(self, image_size, target_std=1.):
        """
        DOG generator class.
        Params:
            sigma1(float): std deviation of the center Gaussian.
            sigma2(float): std of the surround Gaussian
            amp1(float): amplitude of the centre
            amp2(float): amp of the surround
            center (tuple of integers): The position of the filter.
            image_size (tuple of integers): Image height and width.
            target_std:
        Returns:
            2D torch.tensor: A DOG filter.
        """
        
        super().__init__()
        self.sigma1 = nn.Parameter(torch.rand(1)*2, requires_grad=False)
        self.sigma2 = nn.Parameter(torch.rand(1)+2+1, requires_grad=False)
        self.amp1 = nn.Parameter(torch.zeros(1)+1., requires_grad=False)
        self.amp2 = nn.Parameter(torch.zeros(1)+1., requires_grad=False)

        self.center = nn.Parameter(torch.tensor([0., 0.]))
        self.image_size = image_size
        self.target_std = target_std
    
    def forward(self):
        return self.gen_dog()
    
    def gen_dog(self):
        
        # clip values in reasonable range
        self.theta.data.clamp_(-torch.pi, torch.pi)
        self.sigma1.data.clamp_(2., min(self.image_size)/2) #min(self.image_size)/7, min(self.image_size)/5) #2)
        self.sigma2.data.clamp_(2., min(self.image_size)/2) #min(self.image_size)/7, min(self.image_size)/5) #2)

        ymax, xmax = self.image_size
        xmax, ymax = (xmax - 1)/2, (ymax - 1)/2
        xmin = -xmax
        ymin = -ymax
        (y, x) = torch.meshgrid(torch.arange(ymin, ymax+1), torch.arange(xmin, xmax+1))
        dog = torch.exp(-.5 * (x ** 2 / sigma1 ** 2 + y ** 2 / sigma1 ** 2)) - torch.exp(-.5 * (x ** 2 / sigma2 ** 2 + y ** 2 / sigma ** 2))

        return dog.view(1, 1, *self.image_size)
    
    def apply_changes(self):
        self.sigma1.requires_grad_(True)
        self.sigma2.requires_grad_(True)