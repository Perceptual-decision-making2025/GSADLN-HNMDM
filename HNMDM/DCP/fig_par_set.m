function  fig_par_set()

global blks  hlist_block  hlist_par

blks = find_system(gcs, 'Type', 'block');
figure('position',[300 200 500 400],'menubar','none','name','Select the module and variable name');

hlist_block = uicontrol(gcf,'style','listbox',...
               'string',blks,...
               'position',[50 80 200 300],...
               'callback',@listcall_block );
           

diagpar = get_param( blks{1}, 'DialogParameters');      
parnamelist = fieldnames(diagpar);                      
hlist_par = uicontrol(gcf,'style','listbox',...
               'string',parnamelist,...
               'position',[250 80 200 300]);
           
hbutton_ok = uicontrol(gcf,'style','pushbutton',...
               'string','OK',...
               'position',[180 20 60 40],...
               'callback',@buttoncall_ok);
           
hbutton_cancel = uicontrol(gcf,'style','pushbutton',...
               'string','Cancel',...
               'position',[280 20 60 40],...
               'callback',@buttoncall_cancel);
 
end
 

function listcall_block(src,eventdata)

    global blks parnamelist hlist_par

    s = get( gco,'value' ); 
    
    diagpar = get_param( blks{s}, 'DialogParameters');     
    parnamelist = fieldnames(diagpar);                     
    
    set( hlist_par,'string',parnamelist );                  
end

function buttoncall_ok(src,eventdata)
   
    global blks hlist_block parnamelist hlist_par
    
    s = get( hlist_block,'value' ); 
    str = blks{s};
    str = str_bracket(str,['''''']); 
    set_param(gcb,'block_name',str);  

    s2 = get( hlist_par,'value' );  
    
    str2 = parnamelist{s2};
    str2 = str_bracket(str2,['''''']);  
    set_param(gcb,'par_name',str2);  
    
    close(gcf);
    
end

function buttoncall_cancel(src,eventdata)
   close(gcf);
end

