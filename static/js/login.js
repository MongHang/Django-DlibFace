var vm = new Vue({
    el: '#app',
    // 修改Vue變數的讀取語法，避免和django範本語法衝突
    delimiters: ['[[', ']]'],
    data: {
        error_username: false,
        error_password: false,
       error_username_message: '請輸入5-20個字元的帳號',
       error_password_message: '請輸入8-12位元的密碼',
        username: '',
        password: '',
        remembered: true
    },
    methods: {
        // 檢查帳號
        check_username: function(){
            var re = /^[\u4e00-\u9fa5a-zA-Z0-9_-]{5,20}$/;
          if (re.test(this.username)) {
                this.error_username = false;
            } else {
                this.error_username = true;
            }
        },
       // 檢查密碼
        check_pwd: function(){
            var re = /^[0-9A-Za-z]{8,20}$/;
          if (re.test(this.password)) {
                this.error_password = false;
            } else {
                this.error_password = true;
            }
        },
        // 表單提交
        on_submit: function(){
            this.check_username();
            this.check_pwd();

            if (this.error_username == true || this.error_pwd == true) {
                // 不滿足登錄條件：禁用表單
             window.event.returnValue = false
            }
        },

    }
});