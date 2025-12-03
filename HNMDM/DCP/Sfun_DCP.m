function Sfun_DCP(block)
 
setup(block);  
  
function setup(block)
 
  block.NumInputPorts  = 2;   
  block.NumOutputPorts = 0;   
  
  block.SetPreCompInpPortInfoToDynamic;
  block.SetPreCompOutPortInfoToDynamic;

  block.NumDialogPrms     = 4;             
  block.DialogPrmsTunable = {'Nontunable','Nontunable','Nontunable','Nontunable'};
                                         
                                            
  block.InputPort(1).DatatypeID  = 0;                       
  block.InputPort(1).Complexity  = 'Real';                  
  block.InputPort(1).Dimensions = block.DialogPrm(4).Data;   
  
  block.InputPort(2).DatatypeID  = 0;                   
  block.InputPort(2).Complexity  = 'Real';                
  block.InputPort(2).Dimensions = 1;                        
                                      
  SampleType =  block.DialogPrm(3).Data;
  block.SampleTimes =  SampleType;          
 
  block.SetAccelRunOnTLC(false);
  
  block.RegBlockMethod('Update', @Update);

  block.RegBlockMethod('PostPropagationSetup', @DoPostPropSetup);
  
  block.RegBlockMethod('Start', @Start);


function DoPostPropSetup(block)
  block.NumDworks = 1;                   
  
  block.Dwork(1).Name            = 'par_previous';   
  dime =  block.InputPort(1).Dimensions;             
  len = dime(1) * dime(2);                         
  block.Dwork(1).Dimensions      = len;              
  block.Dwork(1).DatatypeID      =  block.InputPort(1).DatatypeID;      
  block.Dwork(1).Complexity      =  block.InputPort(1).Complexity;    
  block.Dwork(1).UsedAsDiscState = true;          

function Start(block)
  
  block_name = block.DialogPrm(1).Data;     
  par_name = block.DialogPrm(2).Data;       
  str = get_param( block_name,par_name );    
  
  num = evalin( 'base',str );               
                                           
  dime =  block.InputPort(1).Dimensions;

  block.Dwork(1).Data = reshape( num,dime(1)*dime(2),1 ); 
  
  set_param(gcb,'reset','off'); 
  
function Update(block)

input = block.InputPort(1).Data;
flag = [block.Dwork(1).Data ~= input(:)];   

% block.CurrentTime ~= 0  

if block.InputPort(2).Data  &&  any(flag)                
    block_name = block.DialogPrm(1).Data;    
    par_name = block.DialogPrm(2).Data;      
    par_val = block.InputPort(1).Data;        
    

    str = mat2str(par_val);
    
    set_param( block_name,par_name,str );   
    
    dime =  block.InputPort(1).Dimensions;     
    block.Dwork(1).Data = reshape( input,dime(1)*dime(2),1 ); 
end



  

