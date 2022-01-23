function  [data_out] = add_message(data_in,message_in,mapflag)
                    switch  mapflag
                        case 1
                               data_in(1:2:end) = data_in(1:2:end)+ message_in(1:2:end) ;
                               data_in(2:2:end) = data_in(2:2:end)+ message_in(1:2:end) ;
                        case 2
                               data_in(1:3:end) = data_in(1:3:end) + message_in(2:3:end) + message_in(3:3:end) ;
                               data_in(2:3:end) = data_in(2:3:end) + message_in(1:3:end) + message_in(3:3:end) ;
                               data_in(3:3:end) = data_in(3:3:end) + message_in(2:3:end) + message_in(1:3:end) ;
                        case 3
                               data_in(1:4:end) = data_in(1:4:end) + message_in(2:4:end) + message_in(3:4:end) + message_in(4:4:end) ;
                               data_in(2:4:end) = data_in(2:4:end) + message_in(1:4:end) + message_in(3:4:end) + message_in(4:4:end) ;
                               data_in(3:4:end) = data_in(3:4:end) + message_in(2:4:end) + message_in(1:4:end) + message_in(4:4:end) ;
                               data_in(3:4:end) = data_in(4:4:end) + message_in(2:4:end) + message_in(1:4:end) + message_in(3:4:end) ;
                    end