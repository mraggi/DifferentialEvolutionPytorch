import torch
from progress_bar import progress_bar

def individual2population(f):
    return lambda P : torch.stack([f(p) for p in P])

class DifferentialEvolver:
    def __init__(self, f, 
                       initial_pop = None, 
                       pop_size=20, dim = (1,), 
                       proj_to_domain = lambda x : x, 
                       f_for_individuals = False, proj_for_individuals = None,
                       maximize = False,
                       use_cuda = False
                ):
        
        if isinstance(dim,int): dim = (dim,)
        
        if initial_pop is None: P = torch.randn(pop_size, *dim)
        else: P = initial_pop
        
        self.pop_size, *self.dim = P.shape
        
        if proj_for_individuals is None: proj_for_individuals = f_for_individuals

        if f_for_individuals: f = individual2population(f)
        if proj_for_individuals: proj_to_domain = individual2population(proj_to_domain)
        
        self.maximize = maximize
        
        if use_cuda: P = P.cuda()
        
        P = proj_to_domain(P)

        self.idx_prob = (1. - torch.eye(pop_size)).to(P)
        self.cost = f(P).squeeze()
        self.P = P
        self.f = f if not maximize else (lambda x: -f(x)) 
        self.proj_to_domain = proj_to_domain
        
        
        
    def step(self, mut=0.8, crossp=0.7):
        I=torch.multinomial(self.idx_prob,3).T
        A,B,C = self.P[I]
        
        mutants = A + mut*(B - C)
        
        T = (torch.rand_like(self.P) < crossp).float()
        
        candidates = self.proj_to_domain(T*mutants + (1-T)*self.P)
        f_candidates = self.f(candidates).squeeze()
        
        should_replace = (f_candidates <= self.cost)
        
        self.cost = torch.where(should_replace,f_candidates,self.cost)
        
        # adjust dimensions for broadcasting
        S = should_replace.float().view(self.pop_size,*[1 for _ in self.dim]) 
        
        self.P = S*candidates + (1-S)*self.P
            
    def best(self):
        best_cost, best_index = torch.min(self.cost, dim=0)
        if self.maximize:
            best_cost *= -1
            
        return best_cost.item(), self.P[best_index]
    
    
def optimize(f, initial_pop = None, 
                pop_size=20, dim = (1,), 
                mut=0.8, crossp=0.7,  
                epochs=1000, 
                proj_to_domain = lambda x : x, 
                f_for_individuals = False, proj_for_individuals = None, 
                maximize = False,
                use_cuda = False):
    
    
    D = DifferentialEvolver(f=f, 
                            initial_pop=initial_pop,
                            pop_size=pop_size, dim = dim, 
                            proj_to_domain = proj_to_domain, 
                            f_for_individuals = f_for_individuals, 
                            proj_for_individuals = proj_for_individuals,
                            maximize=maximize,
                            use_cuda=use_cuda
                           )
    if isinstance(epochs, int): epochs = range(epochs)
        
    pbar = progress_bar(epochs)
    
    test_each = 20
    
    try:
        i = test_each+1
        
        for _ in pbar:
            i -= 1
            D.step(mut=mut, crossp=crossp)
            
            if i == 0:
                i = test_each
                best_cost, _ = D.best()
                pbar.comment = f"| best cost = {best_cost:.4f}"
            
    except KeyboardInterrupt:
        print("Interrupting! Returning best found so far")
    
    return D.best()
