function doLogout() {
    window.location.href = '/logout';
}

function goToMyPage() {
    window.location.href = '/redirect_mypage';
}

function goToWorkSpace() {
    let user_id = 1; //TODO
    window.location.href = `/workspaces/${user_id}`;
}


export { doLogout, goToMyPage, goToWorkSpace };