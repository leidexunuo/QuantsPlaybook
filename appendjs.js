$(function()
{
    $("select[onchange]").each(function()
    {
        $(this).trigger("change");
    });

    $(document).on("focus", ":text[objectname=charge_code]",function(){focusFunc($(this),'');return false;});  
    $(document).on("blur", ":text[objectname=charge_code]",function(){chargeMatch($(this),'');return false;});
    $(document).on("focus", ":text[objectname=charge_description]",function(){focusFunc($(this),'');return false;});  
    $(document).on("blur", ":text[objectname=charge_description]",function(){if($(this).closest('tr[repeat=true]').find(":text[objectname=charge_code]").val()=="") {chargeMatch($(this),'');}});
        
    var personRequest = null;

    $(document).on("change", "select[objectname=person_id]", function () 
    {
        var person_id = $(this).val();

        if (personRequest && personRequest.readyState !== 4)
        {
            personRequest.abort();
        }

        if (!person_id)
        {
            HideWaitBox();
            return;
        }

        if (person_id) 
        {
                var url = ResolveClientUrl('/aspsoft/PopUpDispatcher')+'?nextPID=addCaseArrest_PA&case_id=[se:case_id]&person_id='+encodeURIComponent(person_id)+'&report_id=[qs:report_id]&template_id=[qs:template_id]&PartialRefresh=listIntakeFormMaster_Charge';
            ShowWaitBox();
            personRequest = $.ajax({url: ResolveClientUrl("~/Include/CommonModule/Remote"),
                data:{action:"GETFIRSTCHARGEDOC", staff_id:[se:staff_id], case_id:[util:case_id], person_id: person_id, charge_doc_type: 'ARREST', rnd:Math.random()},
                success:function(data)
                {
                    if (data && data[0] && data[0].charge_document_id)
                    {
                        window.location.href = url+"&charge_document_id="+encodeURIComponent(data[0].charge_document_id);
                    }
                    else
                    {
                        window.location.href = url;                 
                    }
                },
                error:function(xhr, status)
                {
                    if (status !== "abort")
                    {
                        HideWaitBox();
                        alert("Unable to load charge information. Please try again.");
                    }
                },
                complete:function(xhr)
                {
                    if (personRequest === xhr)
                    {
                        personRequest = null;
                    }
                },
                dataType:'json',
                timeout:15000});
        }
    });
    
    var defendantId =getURLParameter("arresteeid",window.location.href);
    var personId = getURLParameter("person_id",window.location.href);
    if(defendantId){
        $("select[objectname=person_id]").val(defendantId);
        $("[objectname=person_id]").trigger("change");
    }
    if(personId){
        $("select[objectname=person_id]").val(personId);
    }
    
    $("input:submit").on('click', function() {
        if (!checkInput()) {
            HideWaitBox();
            return false;
        }
        var warrantBox = $(":text[objectname=charge_doc_no]", $("table[objectname=addSingleArrest_CDR_PA]"));
        var msg = "";
        for (var b = 0; b < warrantBox.length; b++) {
            var wb = $(warrantBox[b]);
            const $container = wb.closest("table").closest("tr").next();
            const nonEmptyCount = $container.find("input[type='text'][objectname='charge_code']").filter(function() 
            {     
                return this.value.trim() !== '';
            }).length;
            // if(wb.val()=="" && $(":text[objectname=charge_code][value!='']",container).length>0)
            // {
            // msg="The Warrant/Summon field is required.";
            // break;
            // }

            if (wb.val() != "" && nonEmptyCount == 0) {
                msg = "The Charge is required for complaint # " + wb.val();
                break;
            }
        }

        if (msg == "") {
            var nonCDRBox = $(":text[objectname=charge_doc_no]", $("table[objectname=addSingleArrest_nonCDR]"));
            for (var c = 0; c < nonCDRBox.length; c++) {
                var nb = $(nonCDRBox[c]);
                const $container = nb.closest("tr");
                const nonEmptyCount = $container.find("input[type='text'][objectname='charge_code']").filter(function() 
                {     
                    return this.value.trim() !== '';
                }).length;
                if (nb.val() == "" && nonEmptyCount > 0) {
                    msg = "The Traffic Complaint# field is required.";
                    break;
                }

                if (nb.val() != "" && nonEmptyCount == 0) {
                    msg = "The Charge is required for complaint # " + nb.val();
                    break;
                }
            }
        }

        if (msg != "") {
            alert(msg);
            HideWaitBox();
            return false;
        } else {
            HideWaitBox();
            return true;
        }
    });
    
    
    $(document).on("blur", "input[objectname=initial_count]", function (e) 
    {
        //e.preventDefault();
        checkDuplicate(this);
        chargeChargeCount(this);
        
    });
    /*
    var otn = "[db3:select otn_no from contact where person_id='[qs:person_id]' and case_id='[qs:case_id]' and flag_delete=0]";
    if(otn!='No_value'){
        $('input[objectname=otn_no]').val(otn);   
    }
     */
    $(".multisection_cmd_add:last","table[objectname=addSingleArrest_CDR_PA]").closest("td").hide();
    
});

function validateChargeGroup(tableSelector, getChargeContainer) {
    const $table = $(tableSelector);
    if ($table.length === 0) return "";

    let errorMsg = "";

    $table.find("input[type='text'][objectname='charge_doc_no']").each(function () {
        const $docNo = $(this);
        const docNoVal = $docNo.val().trim();
        const $container = getChargeContainer($docNo);

        const $chargeCode = $container.find("input[type='text'][objectname='charge_code']");
        const hasChargeCode = $chargeCode.length > 0 && $chargeCode.val().trim() !== "";

        if (docNoVal !== "" && !hasChargeCode) {
            errorMsg = `The Charge is required for complaint # ${docNoVal}`;
            return false; // break each loop
        }
    });

    return errorMsg;
}

function checkDuplicate(v_obj)
{
    var obj        = $(v_obj);
    var objParent  = obj.closest("table[indicator]");
    var trParent   = obj.closest("tr[repeat]");
    var checkCount = $(v_obj).val();
    var lastValue  = 0;
    var duplicate  = false;
    var dupilcateIndex ;
    
        
    $("tr[repeat]", objParent).not(trParent).each(function (index)
    {
        var count = $("[objectname=initial_count]", $(this)).val();
        
        //Get Maximun number
        if(count >= lastValue)
        {
            lastValue = parseInt(count) + 1;
        }
        
        if(checkCount == count)
        {
            duplicate = true;
            dupilcateIndex = index;
        }
        
    })
    //If input count is duplicate , set dupilcated one to maximun
    if(duplicate == true)
    {
        //alert(dupilcateIndex);
        var duplicateObj = $("tr[repeat]", objParent).not(trParent).eq(dupilcateIndex);
        $("[objectname=initial_count]", duplicateObj).val(lastValue);
    }
    
    //Always set maximun number if input count is empty
    if(checkCount == "" || !$.isNumeric(checkCount))
    {
        $(v_obj).val(lastValue);
    }
    
}

function chargeChargeCount(v_obj)
{
    var obj = $(v_obj);
    var objParent = obj.closest("table[indicator]");
    var i = 1;
    var countArray = [];
    var orderArray = [];
    
    var maxNum = 0;
    $("tr[repeat]", objParent).each(function ()
    {
        var initCnt = $("[objectname=initial_count]", $(this));
        var initCnt_value = initCnt.val();
                
        if (!$.isNumeric(initCnt_value))
        {
            initCnt.val('');
        }   
        else if (parseInt(initCnt_value) > parseInt(maxNum))
        {
            maxNum = initCnt_value;
        }
        
        if(initCnt_value =='')
        {
            maxNum = parseInt(maxNum) + 1;
            $("[objectname=initial_count]", $(this)).val(maxNum);
            initCnt_value = maxNum;
        }
        
        countArray.push(Number(initCnt_value));
        
    });
    
    //order by original count
    orderArray = countArray.sort(function (a, b) {
            return a - b
    });
    
    
    //orderArray.forEach(function(element){
    //  alert(element);
    //});
    
        
    $("tr[repeat]", objParent).each(function ()
    {
        var tr = $(this);
        //$("[objectname=initial_count]", tr).val(i);
        if($("[objectname=initial_count]", tr).val()=='')
        {
            var init_cnt =parseInt(maxNum)+i;
            $("[objectname=initial_count]", tr).val(init_cnt); 
             i++;
        }
        
        if($("[objectname=charge_weight]", tr).val()=='')
        {
            $("[objectname=charge_weight]", tr).val('1');          
        }
        
        var count       = $("[objectname=initial_count]", tr);
        var count_value = count.val();
        var countSort   = orderArray.indexOf(Number(count_value)) + 1;
        $("[objectname=initial_count]", tr).val(countSort);
        
    });
}

function getURLParameter(name, url){
    if (!url) url = location.href;
    name = name.replace(/[\[]/, "\\\[").replace(/[\]]/, "\\\]");
    var regexS = "[\\?&]" + name + "=([^&#]*)";
    var regex = new RegExp(regexS);
    var results = regex.exec(url);
    return results == null ? null : results[1];
}

$('form').submit(function(){
    $("[objectname=charge_doc_no]",$("table[objectname=addSingleArrest_CDR_PA]")).each(function(index){
        if($(this).val()==""&&$("input[objectname=charge_code]",$(this).closest("table").closest("tbody")).val()!="") 
        {
            $(this).val("TEMP_"+Math.random());
        }
    });
});
