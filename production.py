from typing import List, Dict, Callable
from queue import PriorityQueue
from node import * 
from uuid import uuid4

class Production () : 
    
    def __init__ (self, 
        priority: int, 
        pred: str, 
        cond: Callable, 
        scope_modifiers: Callable,
        succ: List, 
        prob: List, 
        pred_kwargs={}
        ) : 
        self.id = uuid4()
        self.priority = priority
        self.pred = pred
        self.cond = cond
        self.scope_modifiers = scope_modifiers
        self.succ = succ
        self.prob = prob
        self.pred_kwargs = pred_kwargs
        
    def __lt__ (self, that) : 
        return self.priority < that.priority
    
    def __eq__ (self, that) : 
        return self.priority == that.priority
    
    def is_terminal (self) : 
        return len(self.prob) == 0 
    
def add_to_pq (pq, productions, node) : 
    for prod in productions :
        if node.id == prod.pred: 
            pq.put((prod, node))
            
def run_derivation (productions, axiom, scope) : 
    root = Node(axiom, scope)
    pq = PriorityQueue() 

    add_to_pq(pq, productions, root)
            
    while not pq.empty() :
        production, node = pq.get() 
        if node.is_active() and production.cond(node) and not production.is_terminal(): 
            
            branch_id = sample_idx(production.prob)
            child_scopes = production.scope_modifiers[branch_id](node.scope)
            succ = production.succ[branch_id]
            
            if len(child_scopes) != len(succ) : 
                assert len(succ) == 1, (
                    f'Length mismatch between scopes and successors, '
                    f'len(scopes) = {len(child_scopes)}, '
                    f'len(succ) = {len(succ)}, '
                    f'production id = {production.id}'
                )
                succ = succ * len(child_scopes)
                            
            for sc, s_id in zip(child_scopes, succ) : 
                child_node = Node(s_id, sc)
                add_links(node, child_node)
                add_to_pq(pq, productions, child_node)
                
    return root
