window.HELP_IMPROVE_VIDEOJS = false;
$(document).ready(function() {
// examples
    $('select').on('click', function () {

        var sep_idx = this.value.indexOf('_');
        var domain_name = this.value.substring(0, sep_idx);
        var desired_cmd_idx = parseInt(this.value.substring(sep_idx + 1));
        var current_cmd_idx = current_cmd_idxs[domain_name];

        // hide current content
        var current_content = $('#content_' + domain_name + "_" + current_cmd_idx.toString());

        if (desired_cmd_idx == current_cmd_idx && current_content.is(":visible")) {
            current_content.hide();
            return;
        }
        current_content.hide();

        // show desired content
        var desired_content = $('#content_' + domain_name + "_" + desired_cmd_idx.toString());
        desired_content.show("slow");

        // set current to desired
        current_cmd_idxs[domain_name] = desired_cmd_idx;
    });


    // general function for xyzheader
    function toggle_options(header_id, options_id) {
        if ($(options_id).is(":visible")) {
            $(options_id).hide();
            // extract task name from header. e.g., #gsm8k_header -> gsm8k
            task_name = header_id.split("_")[0].substring(1);

            console.log("You have selected " + task_name + " as your task.");
            for (var i = 0; i <= 100; i++) {

                var content_id = "#content_" + task_name + "_" + i.toString();
                console.log(content_id);
                // check if content exists
                if ($(content_id).length == 0) {
                    break;
                }
                $(content_id).hide();
            }
            $(header_id).removeClass("is-active");
        } else {
            $(options_id).show("slow");
            $(header_id).addClass("is-active");
        }
    }

    $('#rqa1_button').click(function () {
        toggle_options('#rqa1_header', '#rqa1_content');
    });
    $('#rqa2_button').click(function () {
        toggle_options('#rqa2_header', '#rqa2_content');
    });
    $('#rqb1_button').click(function () {
        toggle_options('#rqb1_header', '#rqb1_content');
    });
    $('#rqb2_button').click(function () {
        toggle_options('#rqb2_header', '#rqb2_content');
    });

    $('#cmgtest_button').click(function () {
        toggle_options('#cmgtest_header', '#cmgtest_content');
    });
    $('#llmtest_button').click(function () {
        toggle_options('#llmtest_header', '#llmtest_content');
    });
    $('#filters_button').click(function () {
        toggle_options('#filters_header', '#filters_content');
    });

    $('#filters_filtered_button').click(function () {
        toggle_options('#filters_filtered_header', '#filters_filtered_content');
    });

    $('#filters_rnd_button').click(function () {
        toggle_options('#filters_rnd_header', '#filters_rnd_content');
    });

    $('#filters_oof_button').click(function () {
        toggle_options('#filters_oof_header', '#filters_oof_content');
    });
})