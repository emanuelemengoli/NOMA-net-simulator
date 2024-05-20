# dynamic-cellular-network-simulator-NOMA
 Dynamic Wireless Cellular Network simulator with NOMA RSM

 ### Bs Input
from a JS file read the BS a create 
- class BS (x, y)
Attributes:
- p_x => can be set by default
- p_in + p_out = p_x
- reset flag => if p_x modified, meaning the cell area need to be re-evaluated
- id
- dict UEs list ==> {id : p_level_label}

BS set
- dictionary being BS(IDs) => BS obj


Constraints:
Max cap on number of users, should it consider the number of user 
### UEs spawning
- dict assigned 'BS_UE'==> specifies at which BS is assigned 
- label => active, inactive
- if inactive => clean the dict assigned
- moving toogle==> reassign the UEs
- on spawning call assign
- on dying call death_process
- label inner, outer
Central Register
- activates paging
#- memorizes SNR table
- memorize dict UE-BS ==> launch a max snr query


## Utility fucntions
- assign(UE) ==> compute argmax SNR $\forall$ BSs, update the dict 'BS_UE' with the new BS
- compute gain(d,g0, alpha, s) ==> 
    - computes the BS gain given a UE distance 
    - make a matrix out of it
    - rember UEs are moving so gain is chaingin
- compute SNR(BS,UE) ==> for allocation
    - given the formula, then computes the gain 
- compute SINR(BS,US) intra BS for inner, outer zone allocation
    - given the forumula call gain function
- upon UEs moving ==> handover as UEs ==> naive solution UE set work on the central register, update the dict of UE and BS
- MOVING, ASSIGN,DEATH
